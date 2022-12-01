# -*- coding: utf-8 -*-
#%matplotlib inline

import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
import cufflinks as cf
import warnings
warnings.filterwarnings("ignore")
import pmdarima as pm
import streamlit as st
from tvDatafeed import TvDatafeed ,Interval
import fbprophet
from fbprophet import Prophet

st.set_page_config(layout="wide")

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://wallpapercave.com/download/bull-bear-wallpapers-wp7802925");
             background-attachment: fixed;
	     background-position: 25% 75%;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

st.title('📈 Model Deployment: Forecasting 📉')
st.sidebar.header('Input Company symbol listed on NSE')

COMPANY = st.sidebar.selectbox("Select Company 1 from list",('NIFTY','BANKNIFTY','3MINDIA','AARTIDRUGS','AARTIIND','AAVAS','ABB','ABCAPITAL','ABFRL','ACC',
							   'ACCELYA','ADANIENT','ADANIGAS','ADANIGREEN','ADANIPORTS','ADANIPOWER','ADANITRANS','ADVENZYMES',
							   'AEGISCHEM','AFFLE','AHLUCONT','AIAENG','AJANTPHARM','AKZOINDIA','ALKEM','ALKYLAMINE','ALLCARGO',
							   'AMARAJABAT','AMBER','AMBUJACEM','APARINDS','APLAPOLLO','APLLTD','APOLLOHOSP','APOLLOTYRE','ARVINDFASN',
							   'ASAHIINDIA','ASHOKA','ASHOKLEY','ASIANPAINT','ASTERDM','ASTRAL','ASTRAZEN','ATUL','AUBANK','AUROPHARMA',
							   'AVANTIFEED','AXISBANK','BAJAJ-AUTO','BAJAJCON','BAJAJELEC','BAJAJFINSV','BAJAJHLDNG','BAJFINANCE',
							   'BALKRISIND','BALMLAWRIE','BALRAMCHIN','BANDHANBNK','BANKBARODA','BANKINDIA','BASF','BATAINDIA',
							   'BBTC','BDL','BEL','BEML','BERGEPAINT','BHARATFORG','BHARATRAS','BHARTIARTL','BHEL','BIOCON',
							   'BIRLACORPN','BLUEDART','BLUESTARCO','BOSCHLTD','BPCL','BRIGADE','BRITANNIA','BSE','BSOFT','CADILAHC',
							   'CANBK','CANFINHOME','CAPLIPOINT','CARBORUNIV','CASTROLIND','CCL','CDSL','CEATLTD','CENTRALBK',
							   'CENTURYPLY','CENTURYTEX','CERA','CESC','CGCL','CHALET','CHAMBLFERT','CHOLAFIN','CHOLAHLDNG','CIPLA',
							   'COALINDIA','COCHINSHIP','COLPAL','CONCOR','COROMANDEL','CREDITACC','CRISIL','CROMPTON','CSBBANK',
							   'CUB','CUMMINSIND','CYIENT','DABUR','DALBHARAT','DBCORP','DBL','DCBBANK','DCMSHRIRAM','DEEPAKNTR',
							   'DELTACORP','DEN','DHANUKA','DIAMONDYD','DIVISLAB','DIXON','DLF','DMART','DRREDDY','ECLERX','EDELWEISS',
							   'EICHERMOT','EIDPARRY','EIHOTEL','ELGIEQUIP','EMAMILTD','ENDURANCE','ENGINERSIN','EQUITAS','ERIS',
							   'ESABINDIA','ESCORTS','ESSELPACK','EXIDEIND','FACT','FAIRCHEM','FCONSUMER','FDC','FEDERALBNK',
							   'FINCABLES','FINEORG','FINPIPE','FLFL','FLUOROCHEM','FMGOETZE','FORTIS','FRETAIL','FSL','GAEL',
							   'GAIL','GALAXYSURF','GARFIBRES','GEPIL','GESHIP','GET&D','GICRE','GILLETTE','GLAXO','GLENMARK',
							   'GMMPFAUDLR','GMRINFRA','GNFC','GODFRYPHLP','GODREJAGRO','GODREJCP','GODREJIND','GODREJPROP',
							   'GPPL','GRANULES','GRAPHITE','GRASIM','GREAVESCOT','GREENLAM','GRINDWELL','GRSE','GSFC','GSKCONS',
							   'GSPL','GUJALKALI','GUJGASLTD','GULFOILLUB','HAL','HATHWAY','HATSUN','HAVELLS','HCLTECH','HDFC',
							   'HDFCAMC','HDFCBANK','HDFCLIFE','HEG','HEIDELBERG','HEROMOTOCO','HEXAWARE','HFCL','HGINFRA','HINDALCO',
							   'HINDCOPPER','HINDPETRO','HINDUNILVR','HINDZINC','HONAUT','HSCL','HUDCO','IBREALEST','IBULHSGFIN',
							   'IBVENTURES','ICICIBANK','ICICIGI','ICICIPRULI','ICRA','IDBI','IDEA','IDFC','IDFCFIRSTB','IEX','IGL',
							   'IIFL','IIFLWAM','INDHOTEL','INDIACEM','INDIAMART','INDIANB','INDIGO','INDOCO','INDOSTAR','INDUSINDBK',
							   'INFIBEAM','INFRATEL','INFY','INGERRAND','INOXLEISUR','IOB','IOC','IPCALAB','IRB','IRCON','IRCTC','ISEC',
							   'ITC','ITDC','ITI','JAGRAN','JBCHEPHARM','JCHAC','JINDALSAW','JINDALSTEL','JKCEMENT','JKLAKSHMI','JKPAPER',
							   'JMFINANCIL','JSL','JSWENERGY','JSWHL','JSWSTEEL','JUBILANT','JUBLFOOD','JUSTDIAL','JYOTHYLAB',
							   'KAJARIACER','KALPATPOWR','KANSAINER','KARURVYSYA','KEC','KEI','KIOCL','KIRLOSENG','KNRCON','KOTAKBANK',
							   'KPRMILL','KRBL','KSB','KSCL','KTKBANK','L&TFH','LALPATHLAB','LAOPALA','LAURUSLABS','LAXMIMACH',
							   'LEMONTREE','LICHSGFIN','LINDEINDIA','LT','LTI','LTTS','LUPIN','LUXIND','M&M','M&MFIN','MAHABANK',
							   'MAHINDCIE','MAHLOG','MAHSCOOTER','MAHSEAMLES','MANAPPURAM','MARICO','MARUTI','MASFIN','MAXINDIA',
							   'MCDOWELL-N','METROPOLIS','MFSL','MGL','MHRIL','MIDHANI','MINDACORP','MINDAIND','MINDTREE','MMTC',
							   'MOIL','MOTHERSUMI','MOTILALOFS','MPHASIS','MRF','MRPL','MUTHOOTFIN','NAM-INDIA','NATCOPHARM',
							   'NATIONALUM','NAUKRI','NAVINFLUOR','NAVNETEDUL','NBCC','NCC','NESCO','NETWORK18','NH','NHPC','NIACL',
							   'NIITLTD','NIITTECH','NILKAMAL','NLCINDIA','NMDC','NTPC','OBEROIRLTY','OFSS','OIL','OMAXE','ONGC',
							   'ORIENTELEC','ORIENTREF','PAGEIND','PAPERPROD','PEL','PERSISTENT','PETRONET','PFC','PFIZER','PGHH',
							   'PGHL','PHOENIXLTD','PIDILITIND','PIIND','PNB','PNBHOUSING','PNCINFRA','POLYCAB','POLYMED','POWERGRID',
							   'POWERINDIA','PRESTIGE','PRINCEPIPE','PRSMJOHNSN','PSPPROJECT','PTC','PVR','QUESS','RADICO','RAIN',
							   'RAJESHEXPO','RALLIS','RAMCOCEM','RATNAMANI','RAYMOND','RBLBANK','RCF','RECLTD','REDINGTON','RELAXO',
							   'RELIANCE','RESPONIND','RITES','RVNL','SAIL','SANOFI','SBICARD','SBILIFE','SBIN','SCHAEFFLER','SCHNEIDER',
							   'SCI','SEQUENT','SFL','SHILPAMED','SHOPERSTOP','SHREECEM','SHRIRAMCIT','SIEMENS','SIS','SJVN','SKFINDIA',
							   'SOBHA','SOLARA','SOLARINDS','SONATSOFTW','SPANDANA','SPARC','SRF','SRTRANSFIN','STAR','STARCEMENT',
							   'STRTECH','SUDARSCHEM','SUMICHEM','SUNCLAYLTD','SUNDARMFIN','SUNDRMFAST','SUNPHARMA','SUNTECK','SUNTV',
							   'SUPPETRO','SUPRAJIT','SUPREMEIND','SUVENPHAR','SWANENERGY','SWSOLAR','SYMPHONY','SYNGENE','TASTYBITE',
							   'TATACHEM','TATACOMM','TATACONSUM','TATAELXSI','TATAINVEST','TATAMOTORS','TATAPOWER','TATASTEEL',
							   'TATASTLBSL','TCI','TCIEXP','TCNSBRANDS','TCS','TEAMLEASE','TECHM','TECHNOE','THERMAX','THYROCARE',
							   'TIDEWATER','TIINDIA','TIMKEN','TITAN','TORNTPHARM','TORNTPOWER','TRENT','TRIDENT','TRITURBINE',
							   'TTKPRESTIG','TV18BRDCST','TVSMOTOR','UBL','UCOBANK','UJJIVAN','UJJIVANSFB','ULTRACEMCO','UNIONBANK',
							   'UPL','VAIBHAVGBL','VAKRANGEE','VARROC','VBL','VEDL','VENKEYS','VESUVIUS','VGUARD','VINATIORGA',
							   'VIPIND','VMART','VOLTAS','VRLLOG','VSTIND','VTL','WABCOINDIA','WELCORP','WELSPUNIND','WHIRLPOOL',
							   'WIPRO','WOCKPHARMA','YESBANK','ZEEL','ZENSARTECH','ZYDUSWELL'))

COMPANY1 = st.sidebar.selectbox("Select Company 2 from list",('NIFTY','BANKNIFTY','3MINDIA','AARTIDRUGS','AARTIIND','AAVAS','ABB','ABCAPITAL','ABFRL','ACC',
							   'ACCELYA','ADANIENT','ADANIGAS','ADANIGREEN','ADANIPORTS','ADANIPOWER','ADANITRANS','ADVENZYMES',
							   'AEGISCHEM','AFFLE','AHLUCONT','AIAENG','AJANTPHARM','AKZOINDIA','ALKEM','ALKYLAMINE','ALLCARGO',
							   'AMARAJABAT','AMBER','AMBUJACEM','APARINDS','APLAPOLLO','APLLTD','APOLLOHOSP','APOLLOTYRE','ARVINDFASN',
							   'ASAHIINDIA','ASHOKA','ASHOKLEY','ASIANPAINT','ASTERDM','ASTRAL','ASTRAZEN','ATUL','AUBANK','AUROPHARMA',
							   'AVANTIFEED','AXISBANK','BAJAJ-AUTO','BAJAJCON','BAJAJELEC','BAJAJFINSV','BAJAJHLDNG','BAJFINANCE',
							   'BALKRISIND','BALMLAWRIE','BALRAMCHIN','BANDHANBNK','BANKBARODA','BANKINDIA','BASF','BATAINDIA',
							   'BBTC','BDL','BEL','BEML','BERGEPAINT','BHARATFORG','BHARATRAS','BHARTIARTL','BHEL','BIOCON',
							   'BIRLACORPN','BLUEDART','BLUESTARCO','BOSCHLTD','BPCL','BRIGADE','BRITANNIA','BSE','BSOFT','CADILAHC',
							   'CANBK','CANFINHOME','CAPLIPOINT','CARBORUNIV','CASTROLIND','CCL','CDSL','CEATLTD','CENTRALBK',
							   'CENTURYPLY','CENTURYTEX','CERA','CESC','CGCL','CHALET','CHAMBLFERT','CHOLAFIN','CHOLAHLDNG','CIPLA',
							   'COALINDIA','COCHINSHIP','COLPAL','CONCOR','COROMANDEL','CREDITACC','CRISIL','CROMPTON','CSBBANK',
							   'CUB','CUMMINSIND','CYIENT','DABUR','DALBHARAT','DBCORP','DBL','DCBBANK','DCMSHRIRAM','DEEPAKNTR',
							   'DELTACORP','DEN','DHANUKA','DIAMONDYD','DIVISLAB','DIXON','DLF','DMART','DRREDDY','ECLERX','EDELWEISS',
							   'EICHERMOT','EIDPARRY','EIHOTEL','ELGIEQUIP','EMAMILTD','ENDURANCE','ENGINERSIN','EQUITAS','ERIS',
							   'ESABINDIA','ESCORTS','ESSELPACK','EXIDEIND','FACT','FAIRCHEM','FCONSUMER','FDC','FEDERALBNK',
							   'FINCABLES','FINEORG','FINPIPE','FLFL','FLUOROCHEM','FMGOETZE','FORTIS','FRETAIL','FSL','GAEL',
							   'GAIL','GALAXYSURF','GARFIBRES','GEPIL','GESHIP','GET&D','GICRE','GILLETTE','GLAXO','GLENMARK',
							   'GMMPFAUDLR','GMRINFRA','GNFC','GODFRYPHLP','GODREJAGRO','GODREJCP','GODREJIND','GODREJPROP',
							   'GPPL','GRANULES','GRAPHITE','GRASIM','GREAVESCOT','GREENLAM','GRINDWELL','GRSE','GSFC','GSKCONS',
							   'GSPL','GUJALKALI','GUJGASLTD','GULFOILLUB','HAL','HATHWAY','HATSUN','HAVELLS','HCLTECH','HDFC',
							   'HDFCAMC','HDFCBANK','HDFCLIFE','HEG','HEIDELBERG','HEROMOTOCO','HEXAWARE','HFCL','HGINFRA','HINDALCO',
							   'HINDCOPPER','HINDPETRO','HINDUNILVR','HINDZINC','HONAUT','HSCL','HUDCO','IBREALEST','IBULHSGFIN',
							   'IBVENTURES','ICICIBANK','ICICIGI','ICICIPRULI','ICRA','IDBI','IDEA','IDFC','IDFCFIRSTB','IEX','IGL',
							   'IIFL','IIFLWAM','INDHOTEL','INDIACEM','INDIAMART','INDIANB','INDIGO','INDOCO','INDOSTAR','INDUSINDBK',
							   'INFIBEAM','INFRATEL','INFY','INGERRAND','INOXLEISUR','IOB','IOC','IPCALAB','IRB','IRCON','IRCTC','ISEC',
							   'ITC','ITDC','ITI','JAGRAN','JBCHEPHARM','JCHAC','JINDALSAW','JINDALSTEL','JKCEMENT','JKLAKSHMI','JKPAPER',
							   'JMFINANCIL','JSL','JSWENERGY','JSWHL','JSWSTEEL','JUBILANT','JUBLFOOD','JUSTDIAL','JYOTHYLAB',
							   'KAJARIACER','KALPATPOWR','KANSAINER','KARURVYSYA','KEC','KEI','KIOCL','KIRLOSENG','KNRCON','KOTAKBANK',
							   'KPRMILL','KRBL','KSB','KSCL','KTKBANK','L&TFH','LALPATHLAB','LAOPALA','LAURUSLABS','LAXMIMACH',
							   'LEMONTREE','LICHSGFIN','LINDEINDIA','LT','LTI','LTTS','LUPIN','LUXIND','M&M','M&MFIN','MAHABANK',
							   'MAHINDCIE','MAHLOG','MAHSCOOTER','MAHSEAMLES','MANAPPURAM','MARICO','MARUTI','MASFIN','MAXINDIA',
							   'MCDOWELL-N','METROPOLIS','MFSL','MGL','MHRIL','MIDHANI','MINDACORP','MINDAIND','MINDTREE','MMTC',
							   'MOIL','MOTHERSUMI','MOTILALOFS','MPHASIS','MRF','MRPL','MUTHOOTFIN','NAM-INDIA','NATCOPHARM',
							   'NATIONALUM','NAUKRI','NAVINFLUOR','NAVNETEDUL','NBCC','NCC','NESCO','NETWORK18','NH','NHPC','NIACL',
							   'NIITLTD','NIITTECH','NILKAMAL','NLCINDIA','NMDC','NTPC','OBEROIRLTY','OFSS','OIL','OMAXE','ONGC',
							   'ORIENTELEC','ORIENTREF','PAGEIND','PAPERPROD','PEL','PERSISTENT','PETRONET','PFC','PFIZER','PGHH',
							   'PGHL','PHOENIXLTD','PIDILITIND','PIIND','PNB','PNBHOUSING','PNCINFRA','POLYCAB','POLYMED','POWERGRID',
							   'POWERINDIA','PRESTIGE','PRINCEPIPE','PRSMJOHNSN','PSPPROJECT','PTC','PVR','QUESS','RADICO','RAIN',
							   'RAJESHEXPO','RALLIS','RAMCOCEM','RATNAMANI','RAYMOND','RBLBANK','RCF','RECLTD','REDINGTON','RELAXO',
							   'RELIANCE','RESPONIND','RITES','RVNL','SAIL','SANOFI','SBICARD','SBILIFE','SBIN','SCHAEFFLER','SCHNEIDER',
							   'SCI','SEQUENT','SFL','SHILPAMED','SHOPERSTOP','SHREECEM','SHRIRAMCIT','SIEMENS','SIS','SJVN','SKFINDIA',
							   'SOBHA','SOLARA','SOLARINDS','SONATSOFTW','SPANDANA','SPARC','SRF','SRTRANSFIN','STAR','STARCEMENT',
							   'STRTECH','SUDARSCHEM','SUMICHEM','SUNCLAYLTD','SUNDARMFIN','SUNDRMFAST','SUNPHARMA','SUNTECK','SUNTV',
							   'SUPPETRO','SUPRAJIT','SUPREMEIND','SUVENPHAR','SWANENERGY','SWSOLAR','SYMPHONY','SYNGENE','TASTYBITE',
							   'TATACHEM','TATACOMM','TATACONSUM','TATAELXSI','TATAINVEST','TATAMOTORS','TATAPOWER','TATASTEEL',
							   'TATASTLBSL','TCI','TCIEXP','TCNSBRANDS','TCS','TEAMLEASE','TECHM','TECHNOE','THERMAX','THYROCARE',
							   'TIDEWATER','TIINDIA','TIMKEN','TITAN','TORNTPHARM','TORNTPOWER','TRENT','TRIDENT','TRITURBINE',
							   'TTKPRESTIG','TV18BRDCST','TVSMOTOR','UBL','UCOBANK','UJJIVAN','UJJIVANSFB','ULTRACEMCO','UNIONBANK',
							   'UPL','VAIBHAVGBL','VAKRANGEE','VARROC','VBL','VEDL','VENKEYS','VESUVIUS','VGUARD','VINATIORGA',
							   'VIPIND','VMART','VOLTAS','VRLLOG','VSTIND','VTL','WABCOINDIA','WELCORP','WELSPUNIND','WHIRLPOOL',
							   'WIPRO','WOCKPHARMA','YESBANK','ZEEL','ZENSARTECH','ZYDUSWELL'))


MODEL = st.sidebar.selectbox('Forecasting Model',('Model Based','Data Driven','ARIMA','LSTM Artificial Neural Network','FB Prophet'))

col1, col2 = st.beta_columns((1,1))

#################################################################################
def baseplots(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')
    
    st.subheader('Interactive Candlestick Chart for {}'.format(var))
    fig = cf.Figure(data=[cf.Candlestick(x=data.index,
	   		           	open=data['open'],
			          	high = data['high'],
			           	low = data['low'],
			           	close = data['close'])])
	
    fig.update_layout(xaxis_rangeslider_visible=False,
		      title=f"{var}'s adjusted stock price",
		      xaxis_title="Year",
		      yaxis_title="Stock Price")

    st.plotly_chart(fig, use_container_width = True)


    st.subheader('Line Chart for {}'.format(var))
    fig2 = plt.figure(figsize = (20,8))
    plt.plot(data.close)
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    plt.title('Line Chart')
    plt.grid(True)
    st.write(fig2)

##################################################################################
def model(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    heatmapdata = data[['date','close']]
    heatmapdata['date'] = pd.to_datetime(heatmapdata['date']) 
	# Extracting Day, weekday name, month name, year from the Date column using 
	# Date functions from pandas 
    heatmapdata["month"] = heatmapdata['date'].dt.strftime("%b") # month extraction
    heatmapdata["year"] = heatmapdata['date'].dt.strftime("%Y") # year extraction
    heatmapdata["Day"] = heatmapdata['date'].dt.strftime("%d") # Day extraction
    heatmapdata["wkday"] = heatmapdata['date'].dt.strftime("%A") # weekday extraction

    heatmap_y_month = pd.pivot_table(data = heatmapdata,
                        values = "close",
                        index = "year",
                        columns = "month",
                        aggfunc = "mean",
                        fill_value=0)
    heatmap_y_month1 = heatmap_y_month[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']]

    st.header('Model Based Forecast Result for {}'.format(var))
    st.subheader('Heatmap (Monthly avg)')
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(heatmap_y_month1,annot=True,fmt="g",cmap = 'YlOrBr')
    plt.xlabel('Month')
    plt.ylabel('Year')
    st.pyplot(fig)

    # Boxplot for every month
    st.subheader('Monthly Boxplot')
    fig = plt.figure(figsize=(20,10))
    sns.boxplot(x="month",y="close",data=heatmapdata, order = ["Jan", "Feb","Mar", "Apr","May", "Jun","Jul", "Aug","Sep", "Oct","Nov", "Dec"])
    plt.xlabel('Month')
    plt.ylabel('Stock Price')
    st.pyplot(fig)

    st.subheader('Yearly Boxplot')
    fig = plt.figure(figsize=(20,10))
    sns.boxplot(x="year",y="close",data=heatmapdata)
    sns.lineplot(x="year",y="close",data=heatmapdata)
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    st.pyplot(fig)
	
    sns.lineplot(x="year",y="close",data=heatmapdata)

    #### Splitting data
    data1 = heatmapdata

    data1['t'] = np.arange(1,data1.shape[0]+1)
    data1['t_square'] = np.square(data1.t)
    data1['log_close'] = np.log(data1.close)
    data2 = pd.get_dummies(data1['month'])
    data1 = pd.concat([data1, data2],axis=1)
    data1 = data1.reset_index(drop = True)
    
    # Using 3/4th data for training and remaining for testing
    test_size = round(0.25 * (data1.shape[0]+1))

    Train = data1[:-test_size]
    Test = data1[-test_size:]
    
    ## Trying basic models

    #Linear Model
    import statsmodels.formula.api as smf 

    linear_model = smf.ols('close~t',data=Train).fit()
    pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
    rmse_linear = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_linear))**2))

    #Exponential
    Exp = smf.ols('log_close~t',data=Train).fit()
    pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
    rmse_Exp = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Exp)))**2))

    #Quadratic 
    Quad = smf.ols('close~t+t_square',data=Train).fit()
    pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
    rmse_Quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_Quad))**2))

    #Additive seasonality 
    add_sea = smf.ols('close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
    pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
    rmse_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea))**2))

    #Additive Seasonality Quadratic 
    add_sea_Quad = smf.ols('close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
    pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
    rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['close'])-np.array(pred_add_sea_quad))**2))

    ##Multiplicative Seasonality
    Mul_sea = smf.ols('log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
    pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
    rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_sea)))**2))

    #Multiplicative Additive Seasonality 
    Mul_Add_sea = smf.ols('log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
    pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
    rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['close'])-np.array(np.exp(pred_Mult_add_sea)))**2))

    #Compare the results 
    datamodel = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),
            "RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
    table_rmse=pd.DataFrame(datamodel)
    table = table_rmse.sort_values(['RMSE_Values'],ignore_index = True)

    bestmodel = table.iloc[0,0]

    if bestmodel == "rmse_linear" :
        formula = 'close~t'

    if bestmodel == "rmse_Exp":
        formula = 'log_close~t'

    if bestmodel == "rmse_Quad" :
        formula = 'close~t+t_square'

    if bestmodel == "rmse_add_sea":
        formula = 'close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_add_sea_quad":
        formula = 'close~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_Mult_sea":
        formula = 'log_close~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'

    if bestmodel == "rmse_Mult_add_sea":
        formula = 'log_close~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov'


    #Build the model on entire data set
    model_full = smf.ols(formula,data=data1).fit()
    pred_new  = pd.Series(model_full.predict(data1))

    if bestmodel == ("rmse_Exp" or "rmse_Mult_sea" or "rmse_Mult_add_sea"):
        data1["forecasted_close"] = pd.Series(np.exp(pred_new))
    else:
        data1["forecasted_close"] = pd.Series((pred_new))

    st.subheader('Best Basic Mathematical Model')
    fig = plt.figure(figsize = (20,8))
    plt.xlabel('No.of Days')
    plt.ylabel('Stock Price')
    plt.plot(data1[['close','forecasted_close']].reset_index(drop=True))
    st.pyplot(fig)
	
######################################################################################

def datad(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    import statsmodels.graphics.tsaplots as tsa_plots
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
    from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    heatmapdata = data[['date','close']]

    heatmapdata['date'] = pd.to_datetime(heatmapdata['date']) 
    # Extracting Day, weekday name, month name, year from the Date column using Date functions from pandas 
    heatmapdata["month"] = heatmapdata['date'].dt.strftime("%b") # month extraction
    heatmapdata["year"] = heatmapdata['date'].dt.strftime("%Y") # year extraction
    heatmapdata["Day"] = heatmapdata['date'].dt.strftime("%d") # Day extraction
    heatmapdata["wkday"] = heatmapdata['date'].dt.strftime("%A") # weekday extraction

    data1 = heatmapdata

    data1['t'] = np.arange(1,data1.shape[0]+1)
    data1['t_square'] = np.square(data1.t)
    data1['log_close'] = np.log(data1.close)
    data2 = pd.get_dummies(data1['month'])
    data1 = pd.concat([data1, data2],axis=1)
    data1 = data1.reset_index(drop = True)

    # Using 3/4th data for training and remaining for testing
    test_size = round(0.25 * (data1.shape[0]+1))
    Train = data1[:-test_size]
    Test = data1[-test_size:]

    st.header('Data Driven Forecast Result for {}'.format(var))
    st.subheader('Moving Average(MA)')
    fig = plt.figure(figsize=(20,8))
    data1['close'].plot(label="org")
    for i in range(50,201,50):
        data1["close"].rolling(i).mean().plot(label=str(i))
    plt.xlabel('No.of Days')
    plt.ylabel('Stock Price')
    plt.legend(loc='best')
    st.pyplot(fig)
	
    ### Evaluation Metric RMSE
    def RMSE(pred,org):
        MSE = np.square(np.subtract(org,pred)).mean()   
        return np.sqrt(MSE)

    ### Simple Exponential Method
    ses_model = SimpleExpSmoothing(Train["close"]).fit(smoothing_level=0.2)
    pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
    rmseses = RMSE(pred_ses,Test.close)
    
    ### Holt method
    hw_model = Holt(Train["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
    pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
    rmsehw = RMSE(pred_hw,Test.close)

    ### Holts winter exponential smoothing with additive seasonality and additive trend
    hwe_model_add_add = ExponentialSmoothing(Train["close"],seasonal="add",trend="add",seasonal_periods=365).fit() #add the trend to the model
    pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
    rmsehwaa = RMSE(pred_hwe_add_add,Test.close)

    ### Holts winter exponential smoothing with multiplicative seasonality and additive trend
    hwe_model_mul_add = ExponentialSmoothing(Train["close"],seasonal="mul",trend="add",seasonal_periods=365).fit() 
    pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
    rmsehwma = RMSE(pred_hwe_mul_add,Test.close)

    ### Final Model by combining train and test
    datamodel1 = {"MODEL":pd.Series(["rmse_ses","rmse_hw","rmse_hwe_add_add","rmse_hwe_mul_add"]),"RMSE_Values":pd.Series([rmseses,rmsehw,rmsehwaa,rmsehwma])}

    table_rmse1 = pd.DataFrame(datamodel1)

    table1 = table_rmse1.sort_values(['RMSE_Values'],ignore_index = True)

    bestmodel1 = table1.iloc[0,0]

    if bestmodel1 == "rmse_hwe_add_add" :
        formula1 = ExponentialSmoothing(data["close"],seasonal="add",trend="add",seasonal_periods=365).fit()
    if bestmodel1 == "rmse_hwe_mul_add":
        formula1 = ExponentialSmoothing(data["close"],seasonal="mul",trend="add",seasonal_periods=365).fit()
    if bestmodel1 == "rmse_ses" :
        formula1 = SimpleExpSmoothing(data["close"]).fit(smoothing_level=0.2)
    if bestmodel1 == "rmse_hw":
        formula1 = Holt(data["close"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
    
    #Forecasting for next 12 time periods
    forecasted = formula1.forecast(730)

    st.subheader('Best Holt Winters Model')
    fig = plt.figure(figsize=(20,8))
    plt.plot(data1.close, label = "Actual")
    plt.plot(forecasted, label = "Forecasted")
    plt.xlabel('No.of Days')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(fig)

###########################################################################################################

def arima(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import kpss
	
    st.header('Auto ARIMA Forecast Result for {}'.format(var))
    st.write('**Determining stationarity of the dataset using Augmented Dickey-Fuller Test**')

    result=adfuller (data['close'])
    st.text('Test Statistic: %f' %result[0])
    st.text('p-value: %f' %result[1])

	
    st.write('**Determining stationarity of the dataset using Kwiatkowski Phillips Schmidt Shin (KPSS) test**')
    result_kpss_ct=kpss(data['close'],regression="ct")
    st.text('Test Statistic: %f' %result_kpss_ct[0])
    st.text('p-value: %f' %result_kpss_ct[1])

    st.write('**_Test statistic value greater than 0.05 for both ADFuller and KPSS indicate non-stationarity of the data_**')

    # Auto ARIMA on complete Dataset
    import itertools
    from math import sqrt
    import statsmodels.api as sm
    from sklearn.metrics import mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    ARIMA_model = pm.auto_arima(data['close'], 
                        start_p=1, 
                        start_q=1,
                        test='adf', # use adftest to find optimal 'd'
                        max_p=3, max_q=3, # maximum p and q
                        m=1, # frequency of series (if m==1, seasonal is set to FALSE automatically)
                        d=None,# let model determine 'd'
                        seasonal=False, # No Seasonality for standard ARIMA
                        trace=False, #logs 
                        error_action='warn', #shows errors ('ignore' silences these)
                        suppress_warnings=True,
                        stepwise=True)
    
    from pandas.tseries.frequencies import DAYS
    def forecast(ARIMA_model, periods=730):
        # Forecast
        n_periods = periods
        fitted, confint = ARIMA_model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = pd.date_range(data.index[-1] + pd.DateOffset(days=1), periods = n_periods, freq='D')
        
        # make series for plotting purpose
        fitted_series = pd.Series(fitted.values, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=index_of_fc)
        upper_series = pd.Series(confint[:, 1], index=index_of_fc)

        # Plot
        st.subheader('Auto-ARIMA Forecast')
        fig = plt.figure(figsize=(20,8))
        plt.plot(data["close"])
        plt.plot(fitted_series, color='darkgreen')
        plt.fill_between(lower_series.index, 
                        lower_series, 
                        upper_series, 
                        color='k', alpha=.15)
        plt.title("ARIMA - Forecast of Close Price")
        plt.xlabel('Year')
        plt.ylabel('Stock Price')
        st.pyplot(fig)

    forecast(ARIMA_model)

#############################################################################################

def lstm(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    new_data=data.drop(['symbol','open','high','low','volume','date',],axis=1)

    #creating train and test sets
    dataset = new_data

    test_size = round(0.25 * (dataset.shape[0]+1))
    train = dataset[:-test_size]
    valid = dataset[-test_size:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(46,len(train)):
        x_train.append(scaled_data[i-46:i,0])
        y_train.append(scaled_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    #predicting 896 values, using past 46 from the train data
    inputs = new_data[len(new_data) - len(valid) - 46:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []

    for i in range(46,inputs.shape[0]):
        X_test.append(inputs[i-46:i,0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    # Results
    rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))

    st.header('LSTM Artificial Neural Network based Forecasting for {}'.format(var))
    st.subheader('Forecast by LSTM ANN')
    #for plotting
    fig = plt.figure(figsize=(25,10))
    train = dataset[:-test_size]
    valid = dataset[-test_size:]
    valid['Predictions'] = closing_price
    plt.plot(dataset['close'], label='original')
    plt.plot(valid['Predictions'],label='predicted')
    plt.xlabel('Year')
    plt.ylabel('Stock Price')
    plt.legend()
    st.pyplot(fig)

#################################################################################

def fb(var):
    tv = TvDatafeed()
    data = tv.get_hist(symbol=var,exchange='NSE',n_bars=5000)
    data['date'] = data.index.astype(str)
    new = data['date'].str.split(' ',expand=True)
    data['date'] = new[0]
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date',drop=False)
    
    import fbprophet
    from fbprophet import Prophet

    data2 = data
    data2['ds'] = pd.to_datetime(data['date'])
    data2['y'] = (data2['close'])
    data2 = data2[['ds','y']].reset_index(drop = True)

    model = Prophet()
    model.fit(data2)

    future = model.make_future_dataframe(periods = 730)
    pred = model.predict(future)

    pred.yhat[pred.yhat < 0] = 0
    pred.yhat_lower[pred.yhat_lower < 0] = 0
    pred.yhat_upper[pred.yhat_upper < 0] = 0
    pred.trend_upper[pred.trend_upper < 0] = 0
    pred.trend_lower[pred.trend_lower < 0] = 0

    st.header('Forecast by FB Prophet Model for {}'.format(var))
    st.subheader('Predicted Result')
    st.pyplot(model.plot(pred, xlabel='Year', ylabel='Stock Price'))
    st.subheader('Other Components of FBPROPHET')
    st.write(model.plot_components(pred))

    se = np.square(pred.loc[:, 'yhat'] - data2.y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)

#######################################################################################
with col1:
    baseplots(COMPANY) 

with col2:
    baseplots(COMPANY1)


if MODEL == 'Model Based':
	with col1:
    	    model(COMPANY)
	
	with col2:
    	    model(COMPANY1)
	
if MODEL == 'Data Driven':
	with col1:
    	    datad(COMPANY)
	
	with col2:
    	    datad(COMPANY1)

if MODEL == 'ARIMA':
	with col1:
    	    arima(COMPANY)
	
	with col2:
    	    arima(COMPANY1)

if MODEL == 'LSTM Artificial Neural Network':
	with col1:
    	    lstm(COMPANY)

	with col2:
    	    lstm(COMPANY1)
	
if MODEL == 'FB Prophet':
	with col1:
    	    fb(COMPANY) 

	with col2:
    	    fb(COMPANY1)
