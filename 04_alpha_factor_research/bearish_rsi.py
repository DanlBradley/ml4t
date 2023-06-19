from zipline.api import order_target, record, symbol
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def initialize(context):
    context.i = 0
    context.asset = symbol('AAPL')


def handle_data(context, data):
    # Skip first 300 days to get full windows
    context.i += 1
    if context.i < 252:
        return

    # Compute averages
    # data.history() has to be called with the same params
    # from above and returns a pandas dataframe.
    price = data.history(context.asset, 'price', bar_count=126, frequency="1d")
    result = pd.DataFrame(price)
    result = result.rename(columns={
        result.columns[0]: 'price'
    })
    #result.info()
    #RSI creation block
    result['change'] = result['price'].diff()

    result = result.reset_index()
    change = result.change.tolist()
    rsi = []

    for index, row in result.iterrows():
        rsi_period = 14
        pos_sum = 0
        neg_sum = 0
        count = 0
        for j in range(rsi_period):
            try:
                if change[index - j] >= 0:
                    pos_sum += change[index - j]
                elif change[index - j] < 0:
                    neg_sum -= change[index - j]
                count += 1
            except:
                continue;
        if count == 0:
            rsi.append(np.nan)
        else:
            try:
                rsi_val = 100 - ( 100 / (1 + (pos_sum / neg_sum)))
                rsi.append(rsi_val)
            except:
                print("Div by zero.")
                continue;

    result['rsi'] = pd.DataFrame(rsi)

    #Testing Polyfit function:
    #126 days in a 6-month period
    #RSI
    period = 252
    y_rsi = pd.Series(result[['rsi']].iloc[:,0]).to_numpy()
    x_rsi = result[['rsi']].index.to_numpy()

    # print(x_rsi.shape)
    # print(y_rsi.shape)
    z_rsi = np.polyfit(x_rsi, y_rsi, 2)
    p_rsi = np.poly1d(z_rsi)
    xp_rsi = np.linspace(0, period)

    #Prices
    y = pd.Series(result[['price']].iloc[:,0]).to_numpy()
    x = result[['price']].index.to_numpy()
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    xp = np.linspace(0, period)

    # fig, axes = plt.subplots(2)

    # axes[0].plot(x, y, '.', xp, p(xp))
    # axes[1].plot(x_rsi, y_rsi, '.', xp_rsi, p_rsi(xp_rsi))
    price_trend = round((p(period) - p(0)) / period, 4)
    rsi_trend = round((p_rsi(period) - p_rsi(0)) / period, 2)

    #Trading logic
    if price_trend < 0 and rsi_trend > 0:
        print("Buy condition")
        order_target(context.asset, 1000)
    elif price_trend > 0 and rsi_trend < 0:
        print("Short condition")
        order_target(context.asset, 0)
    elif price_trend > 0 and rsi_trend < 0:
        print("Do nothing")
        order_target(context.asset, 0)

    # Save values for later inspection
    record(AAPL=data.current(context.asset, 'price'),
           price_trend=price_trend,
           rsi_trend=rsi_trend)

def analyze(context, perf):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value in $')
    plt.legend(loc=0)
    plt.show()