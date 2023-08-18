import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import VotingRegressor
#import BaggindRegressor
from sklearn.ensemble import BaggingRegressor

#import linear regression model
from sklearn.linear_model import LinearRegression

#import sequential and dense layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, Conv1D, MaxPooling1D, Flatten

TRAIN_CSV = 'train_2m.csv'
TRAIN_CSV = '12.csv'
ASSET_DETAILS_CSV = 'asset_details.csv'

df_train = pd.read_csv(TRAIN_CSV)

#take the diff of the high price of the next all 10 rows
# df_train['Target'] = df_train['High'].shift(-10) - df_train['High']

#target is the percentage diff of the  high price of the next 10 rows
# df_train['Target'] = df_train['High'].shift(-10) / df_train['High'] - 1

df_train['timestamp'] = pd.to_datetime(df_train['timestamp'], unit='s')

df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")

df_asset_details = df_asset_details[df_asset_details['Asset_ID'].isin(df_train['Asset_ID'].unique())]

class CryptoResearch:

# Two new features from the competition tutorial
    def upper_shadow(self, df):
        return df['High'] - np.maximum(df['Close'], df['Open'])

    def lower_shadow(self, df):
        return np.minimum(df['Close'], df['Open']) - df['Low']

# It works for rows to, so we can reutilize it.
    def get_features(self, df):
        df['Upper_Shadow'] = self.upper_shadow(df)
        df['Lower_Shadow'] = self.lower_shadow(df)

        df['diff_open_close'] = abs((df['Open'] - df['Close'])/df['Open'])
        df['diff_high_low'] = abs((df['High'] - df['Low'])/df['High'])


        df['Hour'] = df['timestamp'].dt.hour
        df['Minute'] = df['timestamp'].dt.minute
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        # df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x > 4 else 0)
        # df['week_of_year'] = df['timestamp'].dt.weekofyear
        # df['week_of_month'] = df['timestamp'].dt.day/7
        df['quarter'] = df['timestamp'].dt.quarter
        df['Month'] = df['timestamp'].dt.month
        # df['is_month_start'] = df['timestamp'].dt.is_month_start
        # df['is_month_end'] = df['timestamp'].dt.is_month_end

        df['day_of_month_sin'] = np.sin(2*np.pi*df['day_of_month']/30)
        df['day_of_month_cos'] = np.cos(2*np.pi*df['day_of_month']/30)
        # df['day_of_year_sin'] = np.sin(2*np.pi*df['day_of_month']/365)
        # df['day_of_year_cos'] = np.cos(2*np.pi*df['day_of_month']/365)

        df['is_night'] = df['Hour'].apply(lambda x: 1 if x > 20 or x < 6 else 0)

        #add the Moving average
        df['MA_5'] = df['Close'].rolling(5).mean()
        df['MA_10'] = df['Close'].rolling(10).mean()
        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()
        df['MA_100'] = df['Close'].rolling(100).mean()
        df['MA_200'] = df['Close'].rolling(200).mean()
        df['MA_500'] = df['Close'].rolling(500).mean()

        #add the Exponential Moving Average
        df['EMA_3'] = df['Close'].ewm(3).mean()
        df['EMA_5'] = df['Close'].ewm(5).mean()
        df['EMA_10'] = df['Close'].ewm(10).mean()
        df['EMA_20'] = df['Close'].ewm(20).mean()
        df['EMA_50'] = df['Close'].ewm(50).mean()
        df['EMA_100'] = df['Close'].ewm(100).mean()
        df['EMA_200'] = df['Close'].ewm(200).mean()
        df['EMA_500'] = df['Close'].ewm(500).mean()

        #add the Moving Average Convergence Divergence
        df['MACD'] = df['EMA_5'] - df['EMA_10']
        df['MACD_signal'] = df['MACD'].ewm(10).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']

        #add the Moving Average Convergence Divergence for 20
        df['MACD_20'] = df['EMA_10'] - df['EMA_20']
        df['MACD_signal_20'] = df['MACD_20'].ewm(10).mean()
        df['MACD_diff_20'] = df['MACD_20'] - df['MACD_signal_20']

        #add the Moving Average Convergence Divergence for 50
        df['MACD_50'] = df['EMA_20'] - df['EMA_50']
        df['MACD_signal_50'] = df['MACD_50'].ewm(10).mean()
        df['MACD_diff_50'] = df['MACD_50'] - df['MACD_signal_50']

        #add the Moving Average Convergence Divergence for 100
        df['MACD_100'] = df['EMA_50'] - df['EMA_100']
        df['MACD_signal_100'] = df['MACD_100'].ewm(10).mean()
        df['MACD_diff_100'] = df['MACD_100'] - df['MACD_signal_100']

        #add the Moving Average Convergence Divergence for 200
        df['MACD_200'] = df['EMA_100'] - df['EMA_200']
        df['MACD_signal_200'] = df['MACD_200'].ewm(10).mean()
        df['MACD_diff_200'] = df['MACD_200'] - df['MACD_signal_200']

        #add the Moving Average Convergence Divergence for 500
        df['MACD_500'] = df['EMA_200'] - df['EMA_500']
        df['MACD_signal_500'] = df['MACD_500'].ewm(10).mean()
        df['MACD_diff_500'] = df['MACD_500'] - df['MACD_signal_500']


        #add the Bollinger Bands
        df['BB_up'] = df['MA_20'] + 2*df['Close'].rolling(20).std()
        df['BB_dn'] = df['MA_20'] - 2*df['Close'].rolling(20).std()

        #add the Bollinger Bands for 50
        df['BB_up_50'] = df['MA_50'] + 2*df['Close'].rolling(50).std()
        df['BB_dn_50'] = df['MA_50'] - 2*df['Close'].rolling(50).std()

        #add the Bollinger Bands for 100
        df['BB_up_100'] = df['MA_100'] + 2*df['Close'].rolling(100).std()
        df['BB_dn_100'] = df['MA_100'] - 2*df['Close'].rolling(100).std()

        #add the Bollinger Bands for 200
        df['BB_up_200'] = df['MA_200'] + 2*df['Close'].rolling(200).std()
        df['BB_dn_200'] = df['MA_200'] - 2*df['Close'].rolling(200).std()

        #add the Bollinger Bands for 500
        df['BB_up_500'] = df['MA_500'] + 2*df['Close'].rolling(500).std()
        df['BB_dn_500'] = df['MA_500'] - 2*df['Close'].rolling(500).std()

        #add the RSI
        delta = df['Close'].diff()
        up_days = delta.copy()
        up_days[delta<=0]=0.0
        down_days = abs(delta.copy())
        down_days[delta>0]=0.0
        RS_up = up_days.rolling(14).mean()
        RS_down = down_days.rolling(14).mean()
        df['RSI'] = 100-100/(1+RS_up/RS_down)

        #add the Stochastic Oscillator
        df['SO%k'] = ((df['Close'] - df['Low'].rolling(14).min()) / (df['High'].rolling(14).max() - df['Low'].rolling(14).min())) * 100
        df['SO%d'] = df['SO%k'].rolling(3).mean()

        #add the Average True Range
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()

        #add the Average True Range for 50
        df['ATR_50'] = (df['High'] - df['Low']).rolling(50).mean()

        #add the Average True Range for 100
        df['ATR_100'] = (df['High'] - df['Low']).rolling(100).mean()

        #add the Average True Range for 200
        df['ATR_200'] = (df['High'] - df['Low']).rolling(200).mean()

        #add the Average True Range for 500
        df['ATR_500'] = (df['High'] - df['Low']).rolling(500).mean()

        #add the Commodity Channel Index
        df['CCI'] = (df['Close'] - df['MA_20']) / (0.015 * df['Close'].rolling(20).std())

        #add the Commodity Channel Index for 50
        df['CCI_50'] = (df['Close'] - df['MA_50']) / (0.015 * df['Close'].rolling(50).std())

        #add the Commodity Channel Index for 100
        df['CCI_100'] = (df['Close'] - df['MA_100']) / (0.015 * df['Close'].rolling(100).std())

        #add the Commodity Channel Index for 200
        df['CCI_200'] = (df['Close'] - df['MA_200']) / (0.015 * df['Close'].rolling(200).std())

        #add the Commodity Channel Index for 500
        df['CCI_500'] = (df['Close'] - df['MA_500']) / (0.015 * df['Close'].rolling(500).std())

        #add the Aroon Indicator
        df['Aroon_up'] = df['High'].rolling(25).apply(lambda x: float(np.argmax(x) + 1) / 25 * 100, raw=True)
        df['Aroon_down'] = df['Low'].rolling(25).apply(lambda x: float(np.argmin(x) + 1) / 25 * 100, raw=True)

        #add the Aroon Indicator for 50
        df['Aroon_up_50'] = df['High'].rolling(50).apply(lambda x: float(np.argmax(x) + 1) / 50 * 100, raw=True)
        df['Aroon_down_50'] = df['Low'].rolling(50).apply(lambda x: float(np.argmin(x) + 1) / 50 * 100, raw=True)

        #add the Aroon Indicator for 100
        df['Aroon_up_100'] = df['High'].rolling(100).apply(lambda x: float(np.argmax(x) + 1) / 100 * 100, raw=True)
        df['Aroon_down_100'] = df['Low'].rolling(100).apply(lambda x: float(np.argmin(x) + 1) / 100 * 100, raw=True)

        #add the Aroon Indicator for 200
        df['Aroon_up_200'] = df['High'].rolling(200).apply(lambda x: float(np.argmax(x) + 1) / 200 * 100, raw=True)
        df['Aroon_down_200'] = df['Low'].rolling(200).apply(lambda x: float(np.argmin(x) + 1) / 200 * 100, raw=True)

        #add the Aroon Indicator for 500
        df['Aroon_up_500'] = df['High'].rolling(500).apply(lambda x: float(np.argmax(x) + 1) / 500 * 100, raw=True)
        df['Aroon_down_500'] = df['Low'].rolling(500).apply(lambda x: float(np.argmin(x) + 1) / 500 * 100, raw=True)

        #add the True Strength Index
        df['TSI'] = df['Close'].diff(1)
        df['TSI'] = df['TSI'].diff(1)
        df['TSI'] = df['TSI'].ewm(25).mean() / df['Close'].diff(1).abs().ewm(25).mean()
        df['TSI_signal'] = df['TSI'].ewm(13).mean()


        #add the True Strength Index for 50
        df['TSI_50'] = df['Close'].diff(1)
        df['TSI_50'] = df['TSI_50'].diff(1)
        df['TSI_50'] = df['TSI_50'].ewm(50).mean() / df['Close'].diff(1).abs().ewm(50).mean()
        df['TSI_signal_50'] = df['TSI_50'].ewm(25).mean()

        #add the True Strength Index for 100
        df['TSI_100'] = df['Close'].diff(1)
        df['TSI_100'] = df['TSI_100'].diff(1)
        df['TSI_100'] = df['TSI_100'].ewm(100).mean() / df['Close'].diff(1).abs().ewm(100).mean()
        df['TSI_signal_100'] = df['TSI_100'].ewm(50).mean()

        #add the True Strength Index for 200
        df['TSI_200'] = df['Close'].diff(1)
        df['TSI_200'] = df['TSI_200'].diff(1)
        df['TSI_200'] = df['TSI_200'].ewm(200).mean() / df['Close'].diff(1).abs().ewm(200).mean()
        df['TSI_signal_200'] = df['TSI_200'].ewm(100).mean()

        #add the True Strength Index for 500
        df['TSI_500'] = df['Close'].diff(1)
        df['TSI_500'] = df['TSI_500'].diff(1)
        df['TSI_500'] = df['TSI_500'].ewm(500).mean() / df['Close'].diff(1).abs().ewm(500).mean()
        df['TSI_signal_500'] = df['TSI_500'].ewm(200).mean()


        #add the Ultimate Oscillator
        df['UO'] = (df['Close'].rolling(7).sum() - df['Close'].rolling(14).sum() + df['Close'].rolling(28).sum()) / (df['High'].rolling(7).sum() - df['Low'].rolling(7).sum() + df['High'].rolling(14).sum() - df['Low'].rolling(14).sum() + df['High'].rolling(28).sum() - df['Low'].rolling(28).sum()) * 100


        #add the Ultimate Oscillator for 14
        df['UO_14'] = (df['Close'].rolling(14).sum() - df['Close'].rolling(28).sum() + df['Close'].rolling(56).sum()) / (df['High'].rolling(14).sum() - df['Low'].rolling(14).sum() + df['High'].rolling(28).sum() - df['Low'].rolling(28).sum() + df['High'].rolling(56).sum() - df['Low'].rolling(56).sum()) * 100

        #add the Ultimate Oscillator for 28
        df['UO_28'] = (df['Close'].rolling(28).sum() - df['Close'].rolling(56).sum() + df['Close'].rolling(112).sum()) / (df['High'].rolling(28).sum() - df['Low'].rolling(28).sum() + df['High'].rolling(56).sum() - df['Low'].rolling(56).sum() + df['High'].rolling(112).sum() - df['Low'].rolling(112).sum()) * 100

        #add the Ultimate Oscillator for 56
        df['UO_56'] = (df['Close'].rolling(56).sum() - df['Close'].rolling(112).sum() + df['Close'].rolling(224).sum()) / (df['High'].rolling(56).sum() - df['Low'].rolling(56).sum() + df['High'].rolling(112).sum() - df['Low'].rolling(112).sum() + df['High'].rolling(224).sum() - df['Low'].rolling(224).sum()) * 100

        #add the Ultimate Oscillator for 112
        df['UO_112'] = (df['Close'].rolling(112).sum() - df['Close'].rolling(224).sum() + df['Close'].rolling(448).sum()) / (df['High'].rolling(112).sum() - df['Low'].rolling(112).sum() + df['High'].rolling(224).sum() - df['Low'].rolling(224).sum() + df['High'].rolling(448).sum() - df['Low'].rolling(448).sum()) * 100

        #add the Chaikin Oscillator
        df['CO'] = df['EMA_3'] - df['EMA_10']
        df['CO_10'] = df['EMA_10'] - df['EMA_20']
        df['CO_20'] = df['EMA_20'] - df['EMA_50']
        df['CO_50'] = df['EMA_50'] - df['EMA_100']
        df['CO_100'] = df['EMA_100'] - df['EMA_200']
        df['CO_200'] = df['EMA_200'] - df['EMA_500']


        #add the Donchian Channel
        df['DC_up'] = df['High'].rolling(20).max()
        df['DC_dn'] = df['Low'].rolling(20).min()
        df['DC_avg'] = (df['DC_up'] + df['DC_dn'])/2

        
        #add the Donchian Channel for 50
        df['DC_up_50'] = df['High'].rolling(50).max()
        df['DC_dn_50'] = df['Low'].rolling(50).min()
        df['DC_avg_50'] = (df['DC_up_50'] + df['DC_dn_50'])/2

        #add the Donchian Channel for 100
        df['DC_up_100'] = df['High'].rolling(100).max()
        df['DC_dn_100'] = df['Low'].rolling(100).min()
        df['DC_avg_100'] = (df['DC_up_100'] + df['DC_dn_100'])/2

        #add the Donchian Channel for 200
        df['DC_up_200'] = df['High'].rolling(200).max()
        df['DC_dn_200'] = df['Low'].rolling(200).min()
        df['DC_avg_200'] = (df['DC_up_200'] + df['DC_dn_200'])/2


        #add the Donchian Channel for 500
        df['DC_up_500'] = df['High'].rolling(500).max()
        df['DC_dn_500'] = df['Low'].rolling(500).min()
        df['DC_avg_500'] = (df['DC_up_500'] + df['DC_dn_500'])/2


        #add the Momentum
        df['MOM'] = df['Close'].diff(10)
        df['MOM_signal'] = df['MOM'].rolling(10).mean()

        df['MOM_20'] = df['Close'].diff(20)
        df['MOM_20_signal'] = df['MOM_20'].rolling(20).mean()

        df['MOM_50'] = df['Close'].diff(50)
        df['MOM_50_signal'] = df['MOM_50'].rolling(50).mean()

        df['MOM_100'] = df['Close'].diff(100)
        df['MOM_100_signal'] = df['MOM_100'].rolling(100).mean()

        df['MOM_200'] = df['Close'].diff(200)
        df['MOM_200_signal'] = df['MOM_200'].rolling(200).mean()

        df['MOM_500'] = df['Close'].diff(500)
        df['MOM_500_signal'] = df['MOM_500'].rolling(500).mean()

        df['MOM_1000'] = df['Close'].diff(1000)
        df['MOM_1000_signal'] = df['MOM_1000'].rolling(1000).mean()

        #add the Rate of Change
        df['ROC'] = df['Close'].pct_change(10)
        df['ROC_20'] = df['Close'].pct_change(20)
        df['ROC_50'] = df['Close'].pct_change(50)
        df['ROC_100'] = df['Close'].pct_change(100)
        df['ROC_200'] = df['Close'].pct_change(200)
        df['ROC_500'] = df['Close'].pct_change(500)
        df['ROC_1000'] = df['Close'].pct_change(1000)

        #add the Daily Return
        df['DR'] = df['Close'].pct_change()
        df['DR_2'] = df['Close'].pct_change(2)
        df['DR_3'] = df['Close'].pct_change(3)
        df['DR_4'] = df['Close'].pct_change(4)


        #add the Daily Log Return
        df['DLR'] = np.log(df['Close']).diff()
        df['DLR_2'] = np.log(df['Close']).diff(2)
        df['DLR_3'] = np.log(df['Close']).diff(3)
        df['DLR_4'] = np.log(df['Close']).diff(4)

        #add the On Balance Volume
        df['OBV'] = df['Volume'].cumsum()

        #add the Accumulation/Distribution Index
        df['ADI'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']

        #add the Chaikin Money Flow
        df['CMF'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']

        #add the Force Index
        df['FI'] = df['Close'].diff(1) * df['Volume']
        df['FI_2'] = df['Close'].diff(2) * df['Volume']
        df['FI_5'] = df['Close'].diff(5) * df['Volume']
        df['FI_10'] = df['Close'].diff(10) * df['Volume']
        df['FI_20'] = df['Close'].diff(20) * df['Volume']
        df['FI_50'] = df['Close'].diff(50) * df['Volume']
        df['FI_100'] = df['Close'].diff(100) * df['Volume']
        df['FI_200'] = df['Close'].diff(200) * df['Volume']
        df['FI_500'] = df['Close'].diff(500) * df['Volume']

        #add the Ease of Movement
        df['EMV'] = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_2'] = (df['High'].diff(2) + df['Low'].diff(2)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_5'] = (df['High'].diff(5) + df['Low'].diff(5)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_10'] = (df['High'].diff(10) + df['Low'].diff(10)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_20'] = (df['High'].diff(20) + df['Low'].diff(20)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_50'] = (df['High'].diff(50) + df['Low'].diff(50)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_100'] = (df['High'].diff(100) + df['Low'].diff(100)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_200'] = (df['High'].diff(200) + df['Low'].diff(200)) * (df['High'] - df['Low']) / (2 * df['Volume'])
        df['EMV_500'] = (df['High'].diff(500) + df['Low'].diff(500)) * (df['High'] - df['Low']) / (2 * df['Volume'])

        #add the Volume Weighted Average Price
        df['VWAP2'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        #add the Keltner Channel
        df['KC_up'] = df['EMA_20'] + 2*df['Close'].rolling(20).std()
        df['KC_dn'] = df['EMA_20'] - 2*df['Close'].rolling(20).std()
        df['KC_avg'] = (df['KC_up'] + df['KC_dn'])/2


        #add the Keltner Channel for the 50
        df['KC_up_50'] = df['EMA_50'] + 2*df['Close'].rolling(50).std()
        df['KC_dn_50'] = df['EMA_50'] - 2*df['Close'].rolling(50).std()
        df['KC_avg_50'] = (df['KC_up_50'] + df['KC_dn_50'])/2

        #add the Keltner Channel for the 100
        df['KC_up_100'] = df['EMA_100'] + 2*df['Close'].rolling(100).std()
        df['KC_dn_100'] = df['EMA_100'] - 2*df['Close'].rolling(100).std()
        df['KC_avg_100'] = (df['KC_up_100'] + df['KC_dn_100'])/2

        #add the Keltner Channel for the 200
        df['KC_up_200'] = df['EMA_200'] + 2*df['Close'].rolling(200).std()
        df['KC_dn_200'] = df['EMA_200'] - 2*df['Close'].rolling(200).std()
        df['KC_avg_200'] = (df['KC_up_200'] + df['KC_dn_200'])/2

        #add the Keltner Channel for the 500
        df['KC_up_500'] = df['EMA_500'] + 2*df['Close'].rolling(500).std()
        df['KC_dn_500'] = df['EMA_500'] - 2*df['Close'].rolling(500).std()
        df['KC_avg_500'] = (df['KC_up_500'] + df['KC_dn_500'])/2


        #add Ichimoku Kinko Hyo
        df['ICHIMOKU_CONV'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min())/2
        df['ICHIMOKU_BASE'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min())/2
        df['ICHIMOKU_A'] = (df['ICHIMOKU_CONV'] + df['ICHIMOKU_BASE'])/2
        df['ICHIMOKU_B'] = (df['High'].rolling(52).max() + df['Low'].rolling(52).min())/2

        #buying and selling signals of the Ichimoku Kinko Hyo
        df['ICHIMOKU_BUY'] = df['ICHIMOKU_CONV'] > df['ICHIMOKU_BASE']

        #add the Parabolic SAR
        df['SAR'] = df['Close'].rolling(2).mean()
        df['SAR_5'] = df['Close'].rolling(5).mean()
        df['SAR_10'] = df['Close'].rolling(10).mean()
        df['SAR_20'] = df['Close'].rolling(20).mean()
        df['SAR_50'] = df['Close'].rolling(50).mean()
        df['SAR_100'] = df['Close'].rolling(100).mean()
        df['SAR_200'] = df['Close'].rolling(200).mean()
        df['SAR_500'] = df['Close'].rolling(500).mean()

        #add Moving average for the volume
        df['VOL_MA_3'] = df['Volume'].rolling(3).mean()
        df['VOL_MA_5'] = df['Volume'].rolling(5).mean()
        df['VOL_MA_10'] = df['Volume'].rolling(10).mean()
        df['VOL_MA_20'] = df['Volume'].rolling(20).mean()
        df['VOL_MA_50'] = df['Volume'].rolling(50).mean()
        df['VOL_MA_100'] = df['Volume'].rolling(100).mean()
        df['VOL_MA_200'] = df['Volume'].rolling(200).mean()
        df['VOL_MA_300'] = df['Volume'].rolling(300).mean()

        #for volume, add the exponential moving average
        df['VOL_EMA_3'] = df['Volume'].ewm(span=3, adjust=False).mean()
        df['VOL_EMA_5'] = df['Volume'].ewm(span=5, adjust=False).mean()
        df['VOL_EMA_10'] = df['Volume'].ewm(span=10, adjust=False).mean()
        df['VOL_EMA_20'] = df['Volume'].ewm(span=20, adjust=False).mean()
        df['VOL_EMA_50'] = df['Volume'].ewm(span=50, adjust=False).mean()
        df['VOL_EMA_100'] = df['Volume'].ewm(span=100, adjust=False).mean()
        df['VOL_EMA_200'] = df['Volume'].ewm(span=200, adjust=False).mean()
        df['VOL_EMA_300'] = df['Volume'].ewm(span=300, adjust=False).mean()

        #add Williams %R
        df['WILLIAMS_%R_3'] = (df['High'].rolling(3).max() - df['Close'])/(df['High'].rolling(3).max() - df['Low'].rolling(3).min())
        df['WILLIAMS_%R_5'] = (df['High'].rolling(5).max() - df['Close'])/(df['High'].rolling(5).max() - df['Low'].rolling(5).min())
        df['WILLIAMS_%R_10'] = (df['High'].rolling(10).max() - df['Close'])/(df['High'].rolling(10).max() - df['Low'].rolling(10).min())
        df['WILLIAMS_%R_20'] = (df['High'].rolling(20).max() - df['Close'])/(df['High'].rolling(20).max() - df['Low'].rolling(20).min())
        df['WILLIAMS_%R_50'] = (df['High'].rolling(50).max() - df['Close'])/(df['High'].rolling(50).max() - df['Low'].rolling(50).min())
        df['WILLIAMS_%R_100'] = (df['High'].rolling(100).max() - df['Close'])/(df['High'].rolling(100).max() - df['Low'].rolling(100).min())
        df['WILLIAMS_%R_200'] = (df['High'].rolling(200).max() - df['Close'])/(df['High'].rolling(200).max() - df['Low'].rolling(200).min())
        df['WILLIAMS_%R_500'] = (df['High'].rolling(500).max() - df['Close'])/(df['High'].rolling(500).max() - df['Low'].rolling(500).min())

        #drop timestamps  and Asset_ID
        df.drop(["timestamp", "Asset_ID"], axis=1, inplace=True)


    def define_keras_model(self, input_shape):

        model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(64, activation='relu'),
            Dense(1),
        ])

        model.compile(loss='mean_squared_error', optimizer='adam')

        return model



    def get_Xy_and_model_for_asset(self, X, asset_id):
        # X = df_train[df_train["Asset_ID"] == asset_id]
        if X.shape[0] == 0:
            return None, None, None, None

        self.get_features(X)
        X.dropna(how="any", inplace=True)

        y = X["Target"]*100000
        X.drop("Target", axis=1, inplace=True)

        #splitting the data into train and test (80% first, 20% last)
        self.X_train = X.iloc[:int(X.shape[0]*0.8)]
        self.X_test = X.iloc[int(X.shape[0]*0.8):]
        self.y_train = y.iloc[:int(y.shape[0]*0.8)]
        self.y_test = y.iloc[int(y.shape[0]*0.8):]

        print('after splitting');

        # lgb_model = LGBMRegressor(n_estimators=25)
        # xgb_model = XGBRegressor(n_estimators=30)
        # linear_model = LinearRegression()
        #use only 50% of the features, and 50% of the instances
        # rf_model = RandomForestRegressor(n_estimators=120, n_jobs=-1, max_features=0.1, max_samples=0.1)

        #add voting regressor with soft voting
        # model = VotingRegressor([('lgb', lgb_model), ('xgb', xgb_model)], n_jobs=-1)
        #use bagging regressor with 50% instances and  50% of features
        # self.model = BaggingRegressor(base_estimator=lgb_model, n_estimators=90, max_samples=0.1, max_features=0.1)
        # self.model = lgb_model
        self.model = self.define_keras_model((self.X_train.shape[1], ))

        print('before fit');
        self.model.fit(self.X_train, self.y_train, epochs=1, batch_size=32, verbose=1)
        print('after fit');

        #evaluating the model

        self.y_pred=self.model.predict(self.X_test)
        rmse=sqrt(mean_squared_error(self.y_test,self.y_pred))
        print("RMSE: ",rmse)


        # trade(model, X_test, y_test)

        #print rmse for every estimator
        # for est in model.estimators_:
            # y_pred=est.predict(X_test)
            # rmse=sqrt(mean_squared_error(y_test,y_pred))
            # print("RMSE: ",rmse)

        # for est in model.estimators_:
            # feature importance
            # features_importances(X_train, est)

        return X, y, self.model, rmse


    def train(self):
        Xs = {}
        ys = {}
        models = {}
        rmses = {}

        for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
            print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
            X, y, model, rmse = self.get_Xy_and_model_for_asset(df_train, asset_id)
            Xs[asset_id], ys[asset_id], models[asset_id], rmses[asset_id] = X, y, model, rmse

        # return model, X_train, y_train, X_test, y_test
        return Xs, ys, models, rmses

    def summary(self, Xs, ys, models, rmses):
        print('minimum for ',df_asset_details.iloc[min(rmses, key=rmses.get)]['Asset_Name'], ' with Value: ', min(rmses.values()))
        print('maximum for ',df_asset_details.iloc[max(rmses, key=rmses.get)]['Asset_Name'], ' with Value: ', max(rmses.values()))
        print('mean: ', np.mean(list(rmses.values())))


    def features_importances(self, X, model):
        # Plot feature importance, sorted from most to least important.
        feature_importance = model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        for feature in reversed(sorted_idx):
            print("{:<20} {:.3f}".format(X.columns[feature], feature_importance[feature]))


    def get_scores(self, model, X, y):
        y_pred = model.predict(X)
        rmse=sqrt(mean_squared_error(y,y_pred))
        print("RMSE: ",rmse)


    def trade(self, model, X, y):

        Y_pred = model.predict(X)

        money = 1000
        coin_size = 0
        buy_limit = 0.000272*100000
        sell_limit = -0.000815*100000
        TRADE_AMOUNT = 100
        buy_times = 0
        sell_times = 0

        buy_X = []
        totals = []

        self.X_clone = X.copy()
        self.X_clone['buy'] = 0
        # cs.X_clone[cs.X_clone.buy == 1].Hour.value_counts()

        #shift Close by 15
        X_Close_shift = X['Close'].shift(-15)
        X_High_shift = X['High'].shift(-15)

        for (i, close, y_pred, x_shift, x_high) in zip(X.index, X['Close'], Y_pred, X_Close_shift, X_High_shift):
            if y_pred > buy_limit and money > 0:
                coin_size = TRADE_AMOUNT / close
                if x_high > close*1.03:
                    sell_price = close*1.03
                else:
                    sell_price = x_high
                money = money - TRADE_AMOUNT + coin_size * sell_price
                buy_times += 1
                totals.append(money)
                #append X to Stats['buy']

                self.X_clone.loc[i, 'buy'] = 1

            # if y_pred < sell_limit and coin_size > 0:
                # money = money + TRADE_AMOUNT
                # coin_size -= TRADE_AMOUNT / close
                # sell_times += 1

        self.totals = totals
        #convert to Dataframe
        # self.buy_X = pd.concat(self.buy_X)
        # totals.append(money + coin_size * close)
        print('buy_times: ', buy_times)
        # print('sell_times: ', sell_times)

        # print("money: ", money)
        # print("coin_size: ", coin_size)
        # print("total: ", money + coin_size * X.iloc[-1]['Close'])
        # print("buy_times: ", buy_times)
        # print("sell_times: ", sell_times)
        # print("Close: ", X.iloc[-1]['Close'])
        return totals
        return money + coin_size * X.iloc[-1]['Close']

    def get_trades(self):
        totals = []
        for size in range(0, len(self.X_test), 1000):
            if(len(self.X_test[:size])) > 1000:
                t = self.trade(self.model, self.X_test[:size], self.y_test[:size])
                totals.append(t)

        return totals

    def plot_trades(self, model, X, y):
        totals = self.trade(model, X, y)
        plt.plot(totals)

    def scores_of_voting_regressor_estimators(self, model, X, y):
        for m in model.estimators_:
            self.get_scores(m, X, y)
