from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import timedelta

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit

def read_dataset(file_storage):
    """Read CSV from uploaded file. Expects columns: date, demand (or quantity)."""
    df = pd.read_csv(file_storage)
    # attempt several common column names
    date_col = None
    for c in df.columns:
        if c.strip().lower() in ['date', 'day', 'ds', 'timestamp']:
            date_col = c
            break
    demand_col = None
    for c in df.columns:
        if c.strip().lower() in ['demand', 'quantity', 'qty', 'value', 'sales']:
            demand_col = c
            break
    if date_col is None or demand_col is None:
        raise ValueError("CSV must contain a date column (date/day/ds) and a demand column (demand/quantity/sales).")
    df = df[[date_col, demand_col]].copy()
    df.columns = ['date', 'demand']
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

def make_time_features(df):
    """Create a simple numeric time feature for regression."""
    df = df.copy()
    df['t'] = (df['date'] - df['date'].min()).dt.days
    # simple seasonality features (weekday)
    df['dow'] = df['date'].dt.weekday
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow')
    df = pd.concat([df, dow_dummies], axis=1)
    return df

def train_forecast_model(df):
    """Train a simple Linear Regression on time + weekday dummies."""
    df2 = make_time_features(df)
    features = ['t'] + [c for c in df2.columns if c.startswith('dow_')]
    X = df2[features].values
    y = df2['demand'].values
    model = LinearRegression()
    model.fit(X, y)
    return model, df2, features

def forecast_next_days(model, df2, features, horizon):
    last_date = df2['date'].max()
    start_t = df2['t'].max()
    preds = []
    pred_dates = []
    for i in range(1, horizon + 1):
        d = last_date + timedelta(days=i)
        t = start_t + i
        dow = d.weekday()
        row = {'t': t}
        # fill dow dummies
        for j in range(7):
            row[f'dow_{j}'] = 1 if j == dow else 0
        Xp = np.array([row[f] for f in features]).reshape(1, -1)
        yhat = model.predict(Xp)[0]
        preds.append(max(0, float(yhat)))  # no negative demand
        pred_dates.append(d)
    forecast_df = pd.DataFrame({'date': pred_dates, 'forecast': preds})
    return forecast_df

def plot_series(history_df, forecast_df):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(history_df['date'], history_df['demand'], label='History', marker='o')
    ax.plot(forecast_df['date'], forecast_df['forecast'], label='Forecast', marker='o')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')
    ax.legend()
    fig.autofmt_xdate()
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return img_b64

def compute_inventory_metrics(history_df, forecast_df, lead_time_days, order_cost, holding_cost_per_unit, service_level):
    """Compute EOQ, safety stock and reorder point using simple formulas.

    - D: annual demand estimated from average daily forecast * 365
    - EOQ = sqrt(2DS / H)
    - Reorder point = (lead time demand) + safety stock
    - Safety stock = z * sigma_lt * sqrt(lead_time_in_periods)
    """
    # estimate daily demand as mean of history + forecast mean
    mean_daily = float(history_df['demand'].mean())
    # if forecast available, blend
    mean_daily = (mean_daily + float(forecast_df['forecast'].mean())) / 2.0
    D = mean_daily * 365.0
    S = max(0.01, float(order_cost))
    H = max(0.0001, float(holding_cost_per_unit))
    # EOQ
    EOQ = np.sqrt(2 * D * S / H)

    # Estimate standard deviation of daily demand from history
    sigma_daily = float(history_df['demand'].std(ddof=0) if history_df['demand'].std(ddof=0) > 0 else 0.001)

    # compute z for service level (normal distribution)
    # approximate z using simple map for common SLs
    z_map = {0.5:0.0, 0.8:0.84, 0.9:1.28, 0.95:1.645, 0.98:2.054, 0.99:2.33}
    try:
        z = z_map.get(round(service_level,2), 1.645)
    except:
        z = 1.645

    lt = max(1, int(lead_time_days))
    safety_stock = z * sigma_daily * np.sqrt(lt)
    # lead time demand
    ltd = mean_daily * lt
    reorder_point = ltd + safety_stock

    metrics = {
        'mean_daily_demand': round(mean_daily,3),
        'annual_demand_estimate': int(round(D)),
        'EOQ': int(round(EOQ)),
        'safety_stock': int(round(safety_stock)),
        'reorder_point': int(round(reorder_point)),
        'lead_time_days': lt
    }
    return metrics

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {
        'error': None,
        'agent_log': [],
        'plot_img': None,
        'metrics': None,
        'forecast_table': None
    }
    if request.method == 'POST':
        try:
            # file upload
            if 'dataset' not in request.files:
                raise ValueError("No file part in request (input name 'dataset').")
            file = request.files['dataset']
            if file.filename == '':
                raise ValueError("No selected file.")
            df = read_dataset(file)
            context['agent_log'].append("Dataset uploaded ({} rows).".format(len(df)))
            # parameters
            horizon = int(request.form.get('horizon', 14))
            lead_time = int(request.form.get('lead_time', 7))
            order_cost = float(request.form.get('order_cost', 50))
            holding_cost = float(request.form.get('holding_cost', 0.5))
            service_level = float(request.form.get('service_level', 0.95))

            context['agent_log'].append(f"Parameters: horizon={horizon}, lead_time={lead_time}, order_cost={order_cost}, holding_cost={holding_cost}, service_level={service_level}")

            # train model
            model, df2, features = train_forecast_model(df)
            context['agent_log'].append("Forecast model trained (LinearRegression). Features: " + ", ".join(features))

            forecast_df = forecast_next_days(model, df2, features, horizon)
            context['agent_log'].append(f"Forecast produced for next {horizon} days.")

            # compute inventory metrics
            metrics = compute_inventory_metrics(df, forecast_df, lead_time, order_cost, holding_cost, service_level)
            context['agent_log'].append("Inventory metrics computed (EOQ, safety stock, reorder point).")

            # plot
            img_b64 = plot_series(df, forecast_df)

            # prepare forecast table_html (top rows)
            forecast_table_html = forecast_df.to_html(index=False, classes='table table-sm', justify='center', border=0)

            context.update({
                'plot_img': img_b64,
                'metrics': metrics,
                'forecast_table': forecast_table_html
            })

        except Exception as e:
            context['error'] = str(e)

    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)
