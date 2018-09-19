
# coding: utf-8

# In[16]:


#Importieren von Bibliotheken
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
from time import *
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


#Ablageort von Dateien
#HIER BITTE ÄNDERN!!!
directory = 'C:/Users/rominger/Documents/rominger/1 BMW Projekt/02 Flexvermarktung/Vermarktung Ladeinfrastruktur/Garching/Intraday-Market/'


# In[18]:


#Einlesen von Ladedaten
file1 = directory + 'EV_Jahresrechnung.csv'
EV_data = pd.read_csv(file1, sep = ";", decimal =",")
EV_data['Start_15'] = pd.to_datetime(EV_data['Start_15'], format = "%d.%m.%Y %H:%M")
EV_data['Ende_15'] = pd.to_datetime(EV_data['Ende_15'], format = "%d.%m.%Y %H:%M" )
EV_data["profits [EUR]"]=0
EV_data["trades [#]"]=0
EV_data["Initialer SOC [kWh]"] = EV_data["Finaler SOC [kWh]"] - EV_data["Energie [kWh]"]


# In[19]:


file2 = directory + 'tx_15_16_17_quarterly_forecast.csv'
# dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%dT%H:%M")
preise = pd.read_csv(file2, sep = ";", decimal =",")
preise['trading_timestamp'] = pd.to_datetime(preise['trading_timestamp'], format = "%Y-%m-%dT%H:%M")
preise = preise.set_index('trading_timestamp')
# Fill missing lines [no price data for any product within 15-min. trading time] with NaN values
preise = preise.resample('15Min').ffill()


# In[ ]:


t1 = clock()
for p in range(210,300):
    
    ##Base data
    #Capacity of battery in kWh
    kap_speicher = EV_data["Kapazitaet [kWh]"][p]
    #Maximum power
    max_power = EV_data['Leistung [kW]'][p]
    #Minimum power 
    min_power = -EV_data['Leistung [kW]'][p]
    #Initial soc
    init_speicher = EV_data['Initialer SOC [kWh]'][p]
    #Final soc
    fSOC = EV_data["Finaler SOC [kWh]"][p]
    #Lead time [15-min.-intervalls]
    leadtime=12*4
    #Horizon of optimization [15-min.-intervalls]
    horizon = int((EV_data['Ende_15'][p]-EV_data['Start_15'][p])/datetime.timedelta(hours=1)*4)
    if horizon <= 2:
        continue
    #charging efficiency
    ladewirkungsgrad = 0.9
    #discharging efficiency
    entladewirkungsgrad = 0.9
    #start time of charge
    start_time = EV_data['Start_15'][p]
    #Calculate time of forecast = time of initial optimization 
    prognosezeitpunkt = start_time - datetime.timedelta(hours=leadtime/4)
    #End Time of charge
    end_time = EV_data['Ende_15'][p] 
    
    ##Create price data for time slot of interest
    #Select Price data from Intraday market for forecast and charge
    price_data=preise.loc[prognosezeitpunkt:end_time]
    #Corrected Price Data for only the following products for each time slot
    corr_price_data = pd.DataFrame(columns = range(int(horizon)))
    #Forecasting time slot - time before charge- price vector with length of horizon
    for i in range(leadtime):
        #Anzahl der ausgelassenen Preise 
        rem_time = leadtime-i-1
        #Preise während des Ladevorgangs
        l = price_data.iloc[i][rem_time:(rem_time+horizon)]
        l = l.reset_index(drop=True)
        corr_price_data=corr_price_data.append(l)
    #Charging time slot - time during charge - price vector with decreasing length
    for j in range(horizon): 
        l = price_data.iloc[i+j+1][:horizon]
        l =l.reset_index(drop=True)
        corr_price_data=corr_price_data.append(l)
    #Add Matrix that specifies non-existing prices (NAN-values) to consider missing trades
    preis_exis = (1-corr_price_data.isnull()*1)
    preis_exis = preis_exis.reset_index(drop = True)
    preis_exis = preis_exis.T  
    #Fill NAN-Values in order to have values for optimization
    corr_price_data=corr_price_data.fillna(method = 'ffill', axis = 1)
    corr_price_data=corr_price_data.fillna(method = 'bfill', axis = 1)
    corr_price_data = corr_price_data.fillna(0)
    zeitreihe = corr_price_data[-horizon:].copy()
    min_price = corr_price_data.min().min()
    max_price = corr_price_data.max().max()

    ##Initial optimization
    #Data for missing prices
    i = 0
    preis_exis[i].to_csv(directory + 'preis_exis.csv',index_label = "t", header = True)
    #Data Frame for price data for Optimization
    preise_buysell = corr_price_data.iloc[0,:]
    #Prices for sell/buy
    preise_buysell = preise_buysell.reset_index(drop = True)
    preise_buysell = pd.DataFrame({'preise': preise_buysell.values})
    preise_buysell.to_csv(directory + 'buysell_preise.csv',index_label = "t")
    #Energy values that restrict charge (here only used final and initial SOC)
    energy_need = preise_buysell.copy()
    energy_need[:] = 0
    energy_need= energy_need.rename(columns ={'preise': 'energy'})
    energy_need['energy'][horizon-1] =fSOC
    energy_need['energy'][0]=init_speicher
    energy_need.to_csv(directory + 'energy_need.csv',index_label = "t")
    #Power values charge = existing schedule for charging
    p_supply = preise_buysell.copy()
    p_supply[:] = 0
    p_supply= p_supply.rename(columns ={'preise': 'Power'})
    p_supply['Power'][:] =0
    p_supply.to_csv(directory + 'p_supply.csv',index_label = "t")
    #Power values discharging = existing schedule for discharging
    p_withdraw = preise_buysell.copy()
    p_withdraw[:] = 0
    p_withdraw= p_withdraw.rename(columns ={'preise': 'Power'})
    p_withdraw.to_csv(directory + 'p_withdraw.csv',index_label = "t")
    

    #Definition of optimization model
    model = AbstractModel()

    #Sets and parameters of the abstract model
    #Sets = Indices
    model.t = Set(dimen = 1) #time periods for trade
    #Parameter = exogenous variables (input variables)
    model.preis_buysell = Param(model.t)
    model.energy_need = Param(model.t)
    model.p_supply = Param(model.t)
    model.p_withdraw = Param(model.t)
    model.preis_exis = Param(model.t)

    #Variables of the abstract model (= decision variables)
    model.p_buy = Var(model.t, domain = Reals, bounds = (0, max_power-min_power), initialize=0)
    model.p_sell = Var(model.t, domain = Reals, bounds = (min_power-max_power, 0), initialize=0)
    model.soc = Var(model.t, domain= NonNegativeReals, bounds = (0, kap_speicher), initialize = 0)
    model.p_result_pos = Var(model.t, domain = Reals, bounds = (0, max_power), initialize=0)
    model.p_result_neg = Var(model.t, domain = Reals, bounds = (min_power, 0), initialize=0)
    model.buy =Var(model.t, domain = Binary)
    model.sell =Var(model.t, domain = Binary)

    #Objective function of the abstract model
    def obj_expression(model):
        return 1/4*1/1000*sum(model.p_buy[t]*model.preis_buysell[t] + model.p_sell[t]*model.preis_buysell[t] for t in model.t)
    model.OBJ = Objective(rule=obj_expression)

    #Schedule of EV 
    def resulting_power_rule(model,t):
        return model.p_result_pos[t] + model.p_result_neg[t]  == model.p_supply[t]+model.p_buy[t]+model.p_withdraw[t]+model.p_sell[t]
    model.resulting_power_rule = Constraint(model.t, rule=resulting_power_rule)

    #Binary variable buy to avoid buying and selling in same time step
    def buy_rule(model,t):
        if t == 0:
            return model.p_buy[t] == 0
        else:
            return model.p_buy[t] <= model.buy[t]*(max_power-min_power)
    model.buy_rule = Constraint(model.t, rule=buy_rule)

    #Binary variable sell to avoid buying and selling in same time step
    def sell_rule(model,t):
        if t == 0:
            return model.p_sell[t] == 0
        else:
            return model.p_sell[t] >= model.sell[t]*(min_power-max_power)
    model.sell_rule = Constraint(model.t, rule=sell_rule)
    
    #binary variable buy II to avoid trading if no trade existed
    def buy_rule2(model,t):
        return model.p_buy[t] <= model.preis_exis[t]*(max_power-min_power)
    model.buy_rule2 = Constraint(model.t, rule=buy_rule2)

    #binary variable sell II to avoid trading if no trade existed
    def sell_rule2(model,t):
        return model.p_sell[t] >= model.preis_exis[t]*(min_power-max_power)
    model.sell_rule2 = Constraint(model.t, rule=sell_rule2)
    
    #constraint to avoid buying and selling in same time step
    def buysell_rule(model,t):
        return model.buy[t]+model.sell[t] <= 1
    model.buysell_rule = Constraint(model.t, rule=buysell_rule)

    #EV SOC
    def soc_rule(model,t):
        if t == 0:
            return model.soc[t] == model.energy_need[t]
        if t >= 1 and t <= horizon:
            return model.soc[t] == model.soc[t-1]+1/4*model.p_result_pos[t]*ladewirkungsgrad+1/4*model.p_result_neg[t]/entladewirkungsgrad
        return Constraint.Skip
    model.soc_rule = Constraint(model.t, rule=soc_rule)

    #MIN SOC
    def min_soc_rule(model,t):
        return model.soc[t] >= model.energy_need[t]
    model.min_soc_rule = Constraint(model.t, rule=min_soc_rule)
    
    
    #Open DataPortal 
    data = DataPortal() 

    #Read all the data from different files
    data.load(filename='buysell_preise.csv',format='set', set='t')
    data.load(filename='buysell_preise.csv',index='t', param='preis_buysell')
    data.load(filename='energy_need.csv',index='t', param='energy_need')
    data.load(filename='p_supply.csv',index='t', param='p_supply')
    data.load(filename='p_withdraw.csv',index='t', param='p_withdraw')
    data.load(filename='preis_exis.csv', index = 't', param = 'preis_exis')
    instance = model.create_instance(data)
    #Use solver gurobi
    opt = SolverFactory('gurobi')
    #Change mipgab to 5%
    opt.options['mipgap'] = 0.05
    model.pprint()
    #Solve
    results = opt.solve(instance) 
    
    # Storing of results in CSV
    name = "results_ev_v0.csv"
    f = open(name, 'w')
    f.write("t" + ", ")
    for t in instance.t.value:
        f.write(str(t)+", ")
    f.write("\n")
    f.write("Power Charge [kW]"+", ")
    for t in instance.t.value:
        f.write(str(instance.p_supply[t]) + ", ")
    f.write("\n")
    f.write("Power Buy [kW]"+", ")
    for t in instance.t.value:
        f.write(str(instance.p_buy[t].value) + ", ")
    f.write("\n")
    f.write("Power Discharge [kW]"+", ")
    for t in instance.t.value:
        f.write(str(instance.p_withdraw[t]) + ", ")
    f.write("\n")
    f.write("Power Sell [kW]"+", ")
    for t in instance.t.value:
        f.write(str(instance.p_sell[t].value) + ", ")
    f.write("\n")
    f.write("Resulting Power [kW]"+", ")
    for t in instance.t.value:
        f.write(str(instance.p_result_pos[t].value+ instance.p_result_neg[t].value) + ", ")
    f.write("\n")
    f.write("Energy [kWh]"+", ")
    for t in instance.t.value:
        f.write(str(instance.soc[t].value) + ", ")
    f.write("\n")
    f.write("Current Price [EUR/MWh]"+", ")
    for t in instance.t.value:
        f.write(str(instance.preis_buysell[t]*instance.preis_exis[t]) + ", ")
    f.write("\n")
    f.write("Existierender Preis [1/0]"+", ")
    for t in instance.t.value:
        f.write(str(instance.preis_exis[t]) + ", ")
    f.write("\n")
    f.close()
    
    #Dataframe with results
    results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
    results_power = results_power.T
    results_power = results_power[:-1]
    results_power = results_power.astype('float')
    results_power['Date']= zeitreihe.index.copy()
    results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
    results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
    #Dataframe with power and energy values
    primary = results_power[['Date','Resulting Power [kW] (Old)','Resulting Power [kW]','Energy [kWh]']].copy()
    primary = primary.rename(columns={"Resulting Power [kW]": "Aktueller Fahrplan [kW]", "Resulting Power [kW] (Old)": "Vorheriger Fahrplan [kW]", "Energy [kWh]": "SOC [kWh]"})
    #Dataframe with prices
    secondary = results_power[['Date','Current Price [EUR/MWh]']].copy()
    secondary = secondary.rename(columns={"Current Price [EUR/MWh]": "Preis aktuelle Optimierung [EUR/MWh]"})
    #Dataframe with buy- and sell power
    tertiary = results_power[['Date','Power Buy [kW]', 'Power Sell [kW]']].copy()
    tertiary= tertiary.rename(columns={"Power Buy [kW]": "Einkauf [kW]", "Power Sell [kW]": "Verkauf [kW]"})
    #Plot
    fig, ax1 = plt.subplots()
    ax1 = primary.plot(x = 'Date', figsize=(13,6.5))
    ax2 = secondary.plot(x = 'Date', secondary_y=True, ylim =(min_power,max(kap_speicher, max_power)+1), ax=ax1)
    tertiary.plot(x = 'Date',kind='bar',ax=ax1)
    ax2.set_ylim(min_price,max_price)
    title = "Intitialoptimierung vor dem Ladevorgang um " + str(corr_price_data.index[0])
    plt.title(title)
    ax1.set_xlabel('Ladevorgang', fontsize = 16)
    ax1.set_ylabel('Leistung [kW]/ Energie [kWh]', fontsize = 16)
    ax2.set_ylabel('Preis [€/MWh]', fontsize = 16)
    #Plot von horizontalen Achsen
    ax1.axhline(y = 0, color = "k")
    ax2.axhline(y = 0, color = "k")
    #Legendenplot
    ax1.legend(bbox_to_anchor=(0,1.05,1,0.15), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
    ax2.legend(bbox_to_anchor=(0,1.10,1,0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
    #Speicherung von Graphen
    fname = 'figure' + str("%02d" % i)
    plt.savefig(fname)
    plt.close()
    plt.close()
    #Creation of cost vector
    profits = pd.Series(horizon+leadtime)
    #Calculation of costs
    profits[0] = 0
    #Creation of vector for number of trades
    trades = pd.Series(horizon+leadtime)
    #Calculation of number of trades
    trades[0] = sum(results_power['Power Buy [kW]'] != 0) + sum(results_power['Power Sell [kW]'] != 0)
    #Series for schedule
    fahrplan = pd.Series(horizon)
    #Series for SOC development
    soc = pd.Series(horizon)
    #Change prices for next optimization
    corr_price_data = corr_price_data.iloc[1:,:]
    corr_price_data = corr_price_data.fillna(method='bfill')
    corr_price_data = corr_price_data.fillna(method='ffill')
    corr_price_data = corr_price_data.fillna(0)

    ##Optimization before charge  
    for i in range(1, leadtime):
        energy_need['energy'][0]= init_speicher
        #Update supply power and discharge power
        power = results_power["Resulting Power [kW]"]
        mask1 = power >= 0
        mask2 = power <= 0
        p_supply['Power'][:] = power*mask1
        p_supply.to_csv(directory + 'p_supply.csv',index_label = "t")
        p_withdraw['Power'][:] = power*mask2
        p_withdraw.to_csv(directory + 'p_withdraw.csv',index_label = "t")
        #Update prices
        preise_buysell = corr_price_data.iloc[0,:]
        preise_buysell = preise_buysell.fillna(method='bfill')
        preise_buysell = preise_buysell.fillna(method='ffill')
        preise_buysell = preise_buysell.reset_index(drop = True)
        preise_buysell = pd.DataFrame({'preise': preise_buysell.values})
        preise_buysell.to_csv(directory + 'buysell_preise.csv',index_label = "t")
        energy_need['energy'][horizon+leadtime]=fSOC
        energy_need.to_csv(directory + 'energy_need.csv',index_label = "t")
        preis_exis[i].to_csv(directory + 'preis_exis.csv',index_label = "t", header = True)

        #Open data portal
        data = DataPortal() 

        #Read all the data from different files
        data.load(filename='buysell_preise.csv',format='set', set='t')
        data.load(filename='buysell_preise.csv',index='t', param='preis_buysell')
        data.load(filename='energy_need.csv',index='t', param='energy_need')
        data.load(filename='p_supply.csv',index='t', param='p_supply')
        data.load(filename='p_withdraw.csv',index='t', param='p_withdraw')
        data.load(filename='preis_exis.csv', index = 't', param = 'preis_exis')
        instance = model.create_instance(data)
        opt = SolverFactory('gurobi')
        opt.options['mipgap'] = 0.05
        results = opt.solve(instance) 
        profits[i] = profits[i-1] + min(instance.OBJ(),0)
        trades[i] = sum(results_power['Power Buy [kW]'] != 0) + sum(results_power['Power Sell [kW]'] != 0)
        
        #Save results in CSVs
        name = "results_ev_v" + str(i) + ".csv"
        f = open(name, 'w')
        f.write("t" + ", ")
        for t in instance.t.value:
            f.write(str(t)+", ")
        f.write("\n")
        f.write("Power Charge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_supply[t]) + ", ")
        f.write("\n")
        f.write("Power Buy [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_buy[t].value) + ", ")
        f.write("\n")
        f.write("Power Discharge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_withdraw[t]) + ", ")
        f.write("\n")
        f.write("Power Sell [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_sell[t].value) + ", ")
        f.write("\n")
        f.write("Resulting Power [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_result_pos[t].value + instance.p_result_neg[t].value) + ", ")
        f.write("\n")
        f.write("Energy [kWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.soc[t].value) + ", ")
        f.write("\n")
        f.write("Current Price [EUR/MWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.preis_buysell[t]*instance.preis_exis[t]) + ", ")
        f.write("\n")
        f.write("Existierender Preis [1/0]"+", ")
        for t in instance.t.value:
            f.write(str(instance.preis_exis[t]) + ", ")
        f.write("\n")
        f.close()

        #Prepare data frames to plot
        results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
        results_power=results_power.T
        results_power=results_power[:-1]
        results_power = results_power.astype('float')
        results_power['Date']= zeitreihe.index.copy()
        results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
        results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
        primary = results_power[['Date','Resulting Power [kW] (Old)','Resulting Power [kW]','Energy [kWh]']].copy()
        primary = primary.rename(columns={"Resulting Power [kW]": "Aktueller Fahrplan [kW]", "Resulting Power [kW] (Old)": "Vorheriger Fahrplan [kW]", "Energy [kWh]": "SOC [kWh]"})
        secondary = results_power[['Date','Current Price [EUR/MWh]']].copy()
        secondary = secondary.rename(columns={"Current Price [EUR/MWh]": "Preis aktuelle Optimierung [EUR/MWh]"})
        tertiary = results_power[['Date','Power Buy [kW]', 'Power Sell [kW]']].copy()
        tertiary= tertiary.rename(columns={"Power Buy [kW]": "Einkauf [kW]", "Power Sell [kW]": "Verkauf [kW]"})
        
        #Plot
        fig, ax1 = plt.subplots()
        ax1 = primary.plot(x = 'Date',figsize=(13,6.5))
        ax2 = secondary.plot(x = 'Date', secondary_y=True, ylim =(min_power,max(kap_speicher, max_power)+1), ax=ax1)
        ax2.set_ylim(min_price,max_price)
        tertiary.plot(x = 'Date', kind='bar',ax=ax1)
        title = "Optimierung vor dem Ladevorgang um " + str(corr_price_data.index[0])
        plt.title(title)
        ax1.set_xlabel('Ladevorgang', fontsize = 16)
        ax1.set_ylabel('Leistung [kW]/ Energie [kWh]', fontsize = 16)
        ax2.set_ylabel('Preis [€/MWh]', fontsize = 16)
        ax1.axhline(y = 0, color = "k")
        ax2.axhline(y = 0, color = "k")
        ax1.legend(bbox_to_anchor=(0,1.05,1,0.15), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        ax2.legend(bbox_to_anchor=(0,1.10,1,0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        fname = 'figure' + str("%02d" % i)
        plt.savefig(fname)
        plt.close()
        plt.close()
        #Preparation of price data for next optimization
        corr_price_data = corr_price_data.iloc[1:,:]
        corr_price_data = corr_price_data.fillna(method='bfill')
        corr_price_data = corr_price_data.fillna(method='ffill')
        corr_price_data = corr_price_data.fillna(0)

    ##Optimization during charge
    for j in range(1,horizon):
        energy_need.loc[0,'energy']= results_power["Energy [kWh]"][1]
        preise_supply = preise_supply[:-1]
        p_supply = p_supply[:-1]
        p_withdraw = p_withdraw[:-1]
        preise_buysell = preise_buysell[:-1]
        energy_need = energy_need[:-1]
        #Update supply power and discharge power
        power = results_power["Resulting Power [kW]"][1:]
        mask1 = power >= 0
        mask2 = power <= 0
        p_supply['Power'][:(horizon-j)] = power*mask1
        p_supply.to_csv(directory + 'p_supply.csv',index_label = "t")
        p_withdraw['Power'][:(horizon-j)] = power*mask2
        p_withdraw.to_csv(directory + 'p_withdraw.csv',index_label = "t")
        #Update prices
        preise_buysell = corr_price_data.iloc[0,:(horizon-j)]
        preise_buysell = preise_buysell.reset_index(drop = True)
        preise_buysell = pd.DataFrame({'preise': preise_buysell.values})
        preise_buysell.to_csv(directory + 'buysell_preise.csv',index_label = "t")
        energy_need.loc[horizon-j-1,'energy']=fSOC
        energy_need.to_csv(directory + 'energy_need.csv',index_label = "t")
        preis_exis[i+j][:(horizon-j)].to_csv(directory + 'preis_exis.csv',index_label = "t", header = True)


        #DataPortal geöffnet
        data = DataPortal() 

        #Read all the data from different files
        data.load(filename='buysell_preise.csv',format='set', set='t')
        data.load(filename='buysell_preise.csv',index='t', param='preis_buysell')
        data.load(filename='energy_need.csv',index='t', param='energy_need')
        data.load(filename='p_supply.csv',index='t', param='p_supply')
        data.load(filename='p_withdraw.csv',index='t', param='p_withdraw')
        data.load(filename='preis_exis.csv', index = 't', param = 'preis_exis')
        
        #Solving of optimization problem
        instance = model.create_instance(data)
        opt = SolverFactory('gurobi')
        opt.options['mipgap'] = 0.05
        results = opt.solve(instance) 
        profits[leadtime+j-1] = profits[leadtime + j-2] + min(instance.OBJ(),0)
        trades[leadtime+j-1] = sum(results_power['Power Buy [kW]'] != 0) + sum(results_power['Power Sell [kW]'] != 0)
        fahrplan[j-1] = results_power['Resulting Power [kW]'][0]
        soc[j-1] = results_power['Energy [kWh]'][0]

        name = "results_ev_v" + str(leadtime+j) + ".csv"
        f = open(name, 'w')
        f.write("t" + ", ")
        for t in instance.t.value:
            f.write(str(t)+", ")
        f.write("\n")
        f.write("Power Charge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_supply[t]) + ", ")
        f.write("\n")
        f.write("Power Buy [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_buy[t].value) + ", ")
        f.write("\n")
        f.write("Power Discharge [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_withdraw[t]) + ", ")
        f.write("\n")
        f.write("Power Sell [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_sell[t].value) + ", ")
        f.write("\n")
        f.write("Resulting Power [kW]"+", ")
        for t in instance.t.value:
            f.write(str(instance.p_result_pos[t].value+ instance.p_result_neg[t].value) + ", ")
        f.write("\n")
        f.write("Energy [kWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.soc[t].value) + ", ")
        f.write("\n")
        f.write("Current Price [EUR/MWh]"+", ")
        for t in instance.t.value:
            f.write(str(instance.preis_buysell[t]*instance.preis_exis[t]) + ", ")
        f.write("\n")
        f.write("Existierender Preis [1/0]"+", ")
        for t in instance.t.value:
            f.write(str(instance.preis_exis[t]) + ", ")
        f.write("\n")
        f.close()
        
        #Dataframe mit noch nicht umgesetztem Fahrplan
        results_power = pd.read_csv(name, index_col = 't',  sep = ",", error_bad_lines=False)
        results_power=results_power.T
        results_power=results_power[:-1]
        results_power = results_power.astype('float')
        results_power['Date']= zeitreihe[j:].index.copy()
        results_power['Date']=results_power['Date'].apply(lambda x: x.strftime('%H:%M'))
        results_power['Resulting Power [kW] (Old)']=results_power['Power Charge [kW]']+results_power['Power Discharge [kW]']
        #Dataframe mit umgesetztem Fahrplan
        realized_power = pd.concat([fahrplan, soc], axis = 1)
        realized_power['Date']= zeitreihe[:j].index.copy()
        realized_power['Date']=realized_power['Date'].apply(lambda x: x.strftime('%H:%M'))
        realized_power = realized_power.rename(columns={0: "Resulting Power [kW]", 1: "Energy [kWh]"})
        final_power = pd.concat([realized_power, results_power])
        primary = final_power[['Date','Resulting Power [kW] (Old)','Resulting Power [kW]','Energy [kWh]']].copy()
        primary = primary.rename(columns={"Resulting Power [kW]": "Aktueller Fahrplan [kW]", "Resulting Power [kW] (Old)": "Vorheriger Fahrplan [kW]", "Energy [kWh]": "SOC [kWh]"})
        secondary = final_power[['Date','Current Price [EUR/MWh]']].copy()
        secondary = secondary.rename(columns={"Current Price [EUR/MWh]": "Preis aktuelle Optimierung [EUR/MWh]"})
        tertiary = final_power[['Date','Power Buy [kW]', 'Power Sell [kW]']].copy()
        tertiary= tertiary.rename(columns={"Power Buy [kW]": "Einkauf [kW]", "Power Sell [kW]": "Verkauf [kW]"})
        fig, ax1 = plt.subplots()
        ax1 = primary.plot(x = 'Date',figsize=(13,6.5))
        ax2 = secondary.plot(x = 'Date', secondary_y=True, ylim =(min_power,max(kap_speicher, max_power)+1), ax=ax1)
        tertiary.plot(x = 'Date',kind='bar',ax=ax1)
        ax2.set_ylim(min_price,max_price)
        plt.axvline(x=j, color ='k', linewidth=2)
        title = "Optimierung während des Ladevorgangs um " + str(corr_price_data.index[0])
        plt.title(title)
        ax1.set_xlabel('Ladevorgang', fontsize = 16)
        ax1.set_ylabel('Leistung [kW]/ Energie [kWh]', fontsize = 16)
        ax2.set_ylabel('Preis [€/MWh]', fontsize = 16)
        ax1.axhline(y = 0, color = "k")
        ax2.axhline(y = 0, color = "k")
        ax1.legend(bbox_to_anchor=(0,1.05,1,0.15), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        ax2.legend(bbox_to_anchor=(0,1.10,1,0.1), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
        #plt.xticks(primary.index)
        fname = 'figure' + str("%02d" % (leadtime+j-1))
        plt.savefig(fname)
        plt.close()
        plt.close()
        #Vorbereitung der Preisdaten für die nächste Optimierung
        corr_price_data = corr_price_data.iloc[1:,:]
        corr_price_data = corr_price_data.fillna(method='bfill')
        corr_price_data = corr_price_data.fillna(method='ffill')
        corr_price_data = corr_price_data.fillna(0)
    
    profits.plot()
    plt.xlabel('Iteration', fontsize = 16)
    plt.ylabel('Cost[€]', fontsize = 16)
    plt.savefig('Costs'+str(p))
    plt.close()
    EV_data["profits [EUR]"].iloc[p]=-profits[leadtime + horizon-3]
    EV_data["trades [#]"].iloc[p] = sum(trades)

t2 = clock()
dt = t2 - t1
#display time for run
dt


# In[17]:


sum(price_data.index == prognosezeitpunkt)

