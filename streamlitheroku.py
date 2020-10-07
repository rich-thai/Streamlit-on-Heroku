import streamlit as st
import pandas as pd
# import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
# from nba_defs import draw_court
# import os
from tensorflow import keras
import pickle
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, make_scorer
from sklearn import tree
import xgboost as xgb
# import cv2


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax




st.title('Kobe Shot Selection Analysis')
st.markdown('This is an analysis of all the shots Kobe Bryant has made throughout his career, including missed shots. This is identified with the shot_made_flag.')
st.markdown('**Data source**: https://www.kaggle.com/c/kobe-bryant-shot-selection/overview')
st.markdown('**Problem:** There are 5000 shot_made_flag values are missing, so the task is estimate the probability that the shot was a success.')
st.markdown('**Approach:** Clean the dataframe of redundant features, visualize the data for insights, engineer additional features that may be useful, develop a binary classifier for the shot_made_flag probability.')

url = 'https://raw.githubusercontent.com/rich-thai/Streamlit-on-Heroku/master/data.csv'

# @st.cache(allow_output_mutation=False)
def load_data():
    data = pd.read_csv(url, index_col=0).reset_index()
    return data


if st.checkbox('Load raw data (first 100)'):
    st.subheader('Raw data (first 100)')
    data_load_state = st.text('Loading data...')
    df = load_data()
    data_load_state.text("Done! (using st.cache)")
    st.write(df.head(100))
    st.markdown('The shape of the dataframe is: '+str(df.shape))


#--------------------------------------------    
st.subheader('Background')
st.markdown('* 5-time NBA champion.')
st.markdown('* 1346 games played')
st.markdown('* 33643 career points')
st.markdown('* Career high of 81 points in a game against Toronto in 2006')
st.markdown('* 26 game winning shots, 8 buzzer beaters')
st.markdown('* 18 NBA All-Start appearances')
st.markdown('* RIP to Kobe, Gianna and others involved in the accident.') 


    
#--------------------------------------------   
st.subheader('Cleaning the data')
st.markdown('* Kobe only played for the Los Angeles Lakers for his entire career, so the features "team_id" and "team_name" are not needed.')
st.markdown('* The "matchup" feature contains "@" if it is an away game, and "vs" for a home game. Create an "away" binary feature.')
st.markdown('* The "matchup" feature is redundant with "opponent", so this can also be dropped.')
st.markdown('* The game date is not important for a professional athlete. His average performance over a season is more useful.')
st.markdown('* lat is linear with loc_y, and lon is linear with loc_x. Is is not necessary to keep both, so I will drop lat and lon.')
st.markdown('* Sort data by game_date and game_event_id.')
st.markdown('* Drop shot_id, game_id, and game_event_id.')


if st.checkbox('Clean the dataframe (first 5 shown)'):
    # cleaning data
    df['game_date'] = pd.to_datetime(df['game_date'])
    df.sort_values(by=['game_date','game_event_id'], inplace=True)
    df['away'] = df['matchup'].str.contains('@')
    df = df.drop(['team_id','team_name','matchup','game_date','lat','lon','shot_id','game_event_id', 'game_id'], axis=1)
    df['period_minutes_remaining'] = df['minutes_remaining'] + df['seconds_remaining']/60

    st.write(df.head())

#--------------------------------------------   
st.subheader('Visualize Shot Selection')


if st.checkbox('Show Visualizations'):
    # create shot location scatter plot
    shotloc_df = df
    action_df = df['action_type']\
        .value_counts().reset_index()\
        .rename(columns={'index':'action type', 'action_type':'Shots Attempted'})
    st.markdown('Spatial Shot Distribution')
    fig, ax = plt.subplots()
    ax.scatter(shotloc_df[shotloc_df['shot_made_flag']==0]['loc_x'], shotloc_df[shotloc_df['shot_made_flag']==0]['loc_y'], s=2, alpha =0.3, color='r')
    ax.scatter(shotloc_df[shotloc_df['shot_made_flag']==1]['loc_x'], shotloc_df[shotloc_df['shot_made_flag']==1]['loc_y'], s=2, alpha =0.3, color='g')
    plt.ylabel('loc_y')
    plt.xlabel('loc_x')
    plt.xlim(-250, 250)
    plt.ylim(-50, 420)
    ax.set_aspect(500/470)
    ax.legend(['Missed','Made'])
    draw_court(outer_lines=True)
    st.pyplot(fig)



    # create shot location scatter multiplot
    fig, ax = plt.subplots(2,2, figsize=(14,20))
    sns.scatterplot(data=shotloc_df, x='loc_x', y='loc_y',hue='shot_zone_area', ax=ax[0,0])
    sns.scatterplot(data=shotloc_df, x='loc_x', y='loc_y',hue='shot_zone_basic', ax=ax[0,1])
    sns.scatterplot(data=shotloc_df, x='loc_x', y='loc_y',hue='shot_zone_range', ax=ax[1,0])
    sns.scatterplot(data=shotloc_df, x='loc_x', y='loc_y',hue='combined_shot_type', ax=ax[1,1])
    plt.ylabel('loc_y')
    plt.xlabel('loc_x')
    plt.xlim(-250, 250)
    plt.ylim(-50, 420)
    xlim=(-250,250)
    ylim=(-50,420)

    for i in range(0,2):
        for j in range(0,2):
            ax[i,j].set_xlim(xlim)
            ax[i,j].set_ylim(ylim)
            draw_court(ax=ax[i,j],outer_lines=True)
            ax[i,j].set_aspect(500/470)

    st.pyplot(fig)


    st.markdown('Action types on a logarithmic scale. A majority of the shots combine a jump shot.')
    fig, ax = plt.subplots(figsize=(7,10))
    ax.barh(action_df['action type'].values, action_df['Shots Attempted'].values, align='center')
    plt.xscale('log')
    plt.xlabel('Action Type')
    st.pyplot(fig)

    st.markdown('There are not many shots beyond 35ft. It is safe to assume that anything beyond this distance is likely a missed shot.')
    fig, ax = plt.subplots(figsize=(10,5))
    ind= np.arange(40)
    sns.distplot(df[(df['shot_made_flag']==0)| (df['shot_made_flag']==1)]['shot_distance'], bins=ind, kde=False,hist_kws=dict(alpha=1))
    sns.distplot(df[(df['shot_made_flag']==1)]['shot_distance'], bins=ind, kde=False,hist_kws=dict(alpha=1))
    ax.legend(['Missed','Made'])
    plt.title('Stacked Bar Chart')
    plt.xlabel('Shot Distance [ft]')
    plt.ylabel('Shot Attempts')
    st.pyplot(fig)


    st.markdown('Average shots throughout the game. Kobe was trusted to take the last shot in the game.' )
    figure, axes = plt.subplots(2, 2,figsize=(10,6))
    binsize=100
    shotloc_df[shotloc_df['period']==1]['period_minutes_remaining'].hist(bins=binsize, ax=axes[0,0])
    shotloc_df[shotloc_df['period']==2]['period_minutes_remaining'].hist(bins=binsize, ax=axes[0,1])
    shotloc_df[shotloc_df['period']==3]['period_minutes_remaining'].hist(bins=binsize, ax=axes[1,0])
    shotloc_df[shotloc_df['period']==4]['period_minutes_remaining'].hist(bins=binsize, ax=axes[1,1])
    figure.text(0.5, 0.04, 'Minutes Elapsed in the Quarter', ha='center')
    figure.text(0.05, 0.5, 'Average Number of Shots / '+str(np.round(12/binsize*60,1))+'s', va='center', rotation='vertical')
    axes[0,0].legend('1st')
    axes[0,1].legend('2nd')
    axes[1,0].legend('3rd')
    axes[1,1].legend('4th')
    st.pyplot(figure)

#--------------------------------------------  
st.subheader('Additional Cleaning')
st.markdown('Machine learning relies on recognizing patterns from large sets of data. There are some rare shots that can generalized.')
st.markdown('* There are many action_types that were not used often. Set a threshold of 20 shot attempts. An action_type below threshold is labelled as combined_shot_type. Drop the combined_shot_type column.')
st.markdown('* There is not much data for shot_distance>30ft. Set an upper limit at 35ft.')
st.markdown('* The specific location of loc_x and loc_y should not be critical, but the general locations from shot_zone_area, shot_zone_basic, shot_zone_range and shot_distance should be.')
st.markdown('* Kobe is trusted to take the last shot in each period. Perhaps the seconds_remaining feature is only useful when there is less than a minute left. Create a "last_seconds" binary feature with a 10s threshold.')

if st.checkbox('Further clean the dataframe (first 5 shown)'):
    action_count = df['action_type'].value_counts()
    df['action'] = df.apply(lambda x: x['action_type'] if action_count[x['action_type']]>20 else x['combined_shot_type'], axis=1)
    df['shot_distance'] = df['shot_distance'].clip(0,35)
    df['last_seconds'] = (df['period_minutes_remaining']<=10/60)
    df.drop(['action_type','combined_shot_type','loc_x','loc_y','seconds_remaining','period_minutes_remaining'], axis=1, inplace=True)

    st.write(df.head())



#--------------------------------------------  
st.subheader('Machine Learning')
st.markdown('Models are evaluated by log-loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))')

st.markdown('The data is separated into a training (X_train_full, y_train_full) and test set (X_test) based on if the shot_made_flag = null.')
st.markdown('Split the training data into 80% for training (X_train, y_train), 20% for validation (X_valid, y_valid).')
st.markdown('**Strategy:** Use grid search to find optimal hyperparameters and k-fold cross-validation.')
st.markdown('**Models:** Decision tree, Random forest, XGBoost.') 


#--------------------------------------------  
st.subheader('Tree-Based Methods')
st.markdown('* One of the most intuitive types of machine learning models.')
st.markdown('* Finds the best feature and threshold that minimizes the impurity down the tree (gini).')
st.markdown('* Can handle categorical features with ordinal encoding or one-hot encoding.')
st.markdown('* Does not require feature scaling as opposed to other types of models.')

st.markdown('With one-hot encoding and ensemble methods, we can determine which features are important after training the model.')


if st.checkbox('One-hot encode the dataframe (first 5 shown)'):
    dummies_df = pd.get_dummies(df[['minutes_remaining','last_seconds','period', 'playoffs', 'season', 'shot_distance',
           'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
           'opponent', 'action','away']].astype(str))
    df = pd.concat([dummies_df,df['shot_made_flag']],axis=1)

    ## Split into train/test set based on missing value
    X_test = df[df['shot_made_flag'].isnull()].drop('shot_made_flag',axis=1)
    X_train_full = df[df['shot_made_flag'].notnull()]
    y_train_full = X_train_full.pop('shot_made_flag')
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    st.write(X_train.head())




#--------------------------------------------  
st.subheader('Decision Tree Classifier')

param_grid = [
{'max_depth': [2,3,4,5], 'max_features':[2,4,8,16,None], 'max_leaf_nodes':[2,3,4,None]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
dtree = DecisionTreeClassifier(random_state=0)
dtree_state = st.text('Loading decision tree classifier...')
dtreeCV  = pickle.load(open('Models/dtree_model.pkl', 'rb'))
dtree_state.text('Loaded decision tree classifier.')

if st.checkbox('Show decision tree output'):
    st.text(dtreeCV.best_params_)
    st.text([-1*dtreeCV.best_score_, dtreeCV.best_estimator_])


    fig = plt.figure(figsize=(7,7))
    _ = tree.plot_tree(dtreeCV.best_estimator_, 
                       feature_names=X_train_full.columns.tolist(),  
                       class_names=['0','1'],
                       filled=True)
    fig.canvas.draw()
    figdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    figdata = figdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    st.image(figdata)



    fig, ax = plt.subplots(figsize=(10,5))
    plt.hist(dtreeCV.predict_proba(X_test)[:,0], alpha=0.5)
    plt.hist(dtreeCV.predict_proba(X_test)[:,1], alpha=0.5)
    plt.xlabel('X_test Shot Probability')
    plt.ylabel('Counts')
    ax.legend(['Missed','Made'])
    st.pyplot(fig)

st.markdown('We can see that this model is very simplistic and does not predict a variety of shot probabilities.')
st.markdown('Kaggle evaluates this model on the test set and gives a log-loss metric of 0.62.')



#--------------------------------------------  
st.subheader('Random Forest Classifier')
st.markdown('An ensemble of decision trees where subsets of data is divided amongst tree. The ensemble votes for the most likely outcome.')

param_grid = [
{'max_depth': [7], 'max_features':[5,10,15]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
RFclf = RandomForestClassifier(random_state=0, n_estimators=100)
RFclfCV = pickle.load(open('Models/rfclf_model.pkl', 'rb'))

if st.checkbox('Show RF output'):
    st.text(RFclfCV.best_params_)
    st.text([-1*RFclfCV.best_score_, RFclfCV.best_estimator_])

    fig, ax = plt.subplots(figsize=(10,5))
    plt.hist(RFclfCV.predict_proba(X_test)[:,0], alpha=0.5)
    plt.hist(RFclfCV.predict_proba(X_test)[:,1], alpha=0.5)
    plt.xlabel('X_test Shot Probability')
    plt.ylabel('Counts')
    ax.legend(['Missed','Made'])
    st.pyplot(fig)

st.markdown('The log-loss evaluation is slightly worse but there is a larger variety of shot probabilities.')
st.markdown('Kaggle evaluates this model on the test set and gives a log-loss metric of 0.62.')


#--------------------------------------------  
st.subheader('Gradient Boosting with XGBoost')
st.markdown('Decision trees are sequentially fit to the residuals, and outputs are summed.')

xgb_clfCV = pickle.load(open('Models/xgbclf_model.pkl', 'rb'))

if st.checkbox('Show XGBoost output'):
    st.text(RFclfCV.best_params_)
    st.text([-1*xgb_clfCV.best_score_, xgb_clfCV.best_estimator_])


    fig, ax = plt.subplots(figsize=(10,5))
    plt.hist(xgb_clfCV.predict_proba(X_test)[:,0], alpha=0.5)
    plt.hist(xgb_clfCV.predict_proba(X_test)[:,1], alpha=0.5)
    plt.xlabel('X_test Shot Probability')
    plt.ylabel('Counts')
    ax.legend(['Missed','Made'])
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,10))
    pd.Series(xgb_clfCV.best_estimator_.get_booster().get_score()).sort_values(ascending=False).head(20).plot.barh()
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    st.pyplot(fig)

st.markdown('This has the lowest log-loss from 10-fold cross-validation. We can see that playing in a Home game seems to have a large impact on the shot probability.')
st.markdown('Kaggle evaluates this model on the test set and gives a log-loss metric of 0.60.')


#--------------------------------------------  
st.subheader('Kobe Shot Predictor')
if st.checkbox('Show this section'):
    st.markdown('Interactive shot prediction using parameters in the sidebar. To make this run a bit faster, uncheck all outputs above this section.')
    st.markdown('A neural network is trained to predict the shot_type, shot_zone_area, shot_zone_basic, shot_zone_range and most probable action type given loc_x and loc_y.')
    st.markdown('This is trained for all shots (missed and made), but for action_type, it is only trained on shots made to choose the most likely successful shot. This determines the top 5 action_types and action_probability.')
    st.markdown('From this + slidebar parameters, the inputs are fed to the XGBoost model to predict the shot_sucess_probability.')




    ## load models
    loc_scaler = pickle.load(open('Models/loc_scaler.pkl', 'rb'))
    loc_scaler2 = pickle.load(open('Models/loc_scaler2.pkl', 'rb'))
    encoder = pickle.load(open('Models/encoder.pkl', 'rb'))
    encoderMade = pickle.load(open('Models/encoderMade.pkl', 'rb'))
    pointValueModel = keras.models.load_model('Models/pointValueModel')
    zoneAreaModel = keras.models.load_model('Models/zoneAreaModel')
    zoneBasicModel = keras.models.load_model('Models/zoneBasicModel')
    zoneRangeModel = keras.models.load_model('Models/zoneRangeModel')
    actionModel = keras.models.load_model('Models/actionModel')


    teams = ['ATL','BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'MEM', 'MIA', 'MIL', 'MIN', 'NJN', 'NOH', 'NOP','NYK','OKC','ORL','PHI','PHX','POR','SAC', 'SAS', 'SEA', 'TOR', 'UTA', 'VAN', 'WAS']
    seasons = ['1996-97', '1997-98', '1998-99', '1999-00', '2000-01', '2001-02',
           '2002-03', '2003-04', '2004-05', '2005-06', '2006-07', '2007-08',
           '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14',
           '2014-15', '2015-16']

    x_slider = st.sidebar.slider('loc_x',min_value=-250, max_value=250, value=0, step=1)
    y_slider = st.sidebar.slider('loc_y',min_value=-50, max_value=420, value=300, step=1)
    season_slider = st.sidebar.select_slider(
         "Season",
         options=seasons, value='2010-11')
    teams_slider = st.sidebar.select_slider(
         "Opponent",
         options=teams, value='TOR')
    period_slider = st.sidebar.slider('Quarter',min_value=1, max_value=4, value=1, step=1)
    time_slider = st.sidebar.slider('Minutes Remaining',min_value=0, max_value=11, value=11, step=1)
    away_radio = st.sidebar.radio('Away?',('True','False'))


    xysq= np.sqrt(x_slider**2 + y_slider**2)
    const = 10.53425466162496
    coord = [[x_slider, y_slider, xysq]]


    position_df = pd.DataFrame(encoder.inverse_transform(pd.DataFrame([pointValueModel.predict_classes(loc_scaler.transform(coord))
                                                                       ,zoneAreaModel.predict_classes(loc_scaler.transform(coord))
                                                                       ,zoneBasicModel.predict_classes(loc_scaler.transform(coord))
                                                                       ,zoneRangeModel.predict_classes(loc_scaler.transform(coord))]).transpose()),
                              columns=['shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range'])
    shot_est = encoderMade.inverse_transform(actionModel.predict_classes(loc_scaler2.transform(coord)).reshape(-1,1))[0][0]

    st.write('Neural network ouput:')

    st.write(position_df)

    input_df = pd.read_csv('empty_df.csv')

    shot_est2 = pd.DataFrame(actionModel.predict(loc_scaler2.transform(coord)).transpose()).reset_index().rename(columns={'index':'action_type',0:'action_probability'})
    shot_est2['action_type'] = encoderMade.inverse_transform(shot_est2['action_type'].values.reshape(-1,1))
    shot_est2 = shot_est2.sort_values(by='action_probability', ascending=False).head().reset_index().drop('index',axis=1)
    action_dummies_df = pd.get_dummies(shot_est2['action_type'], prefix='action')

    input_df = input_df.merge(action_dummies_df, how='right')
    input_df.fillna(0, inplace=True)

    minutes_remaining = str(time_slider)
    last_seconds='False'
    period=str(period_slider)
    playoffs='0'
    season=season_slider
    shot_distance= str(np.clip(np.int(xysq/const),0,35))
    shot_zone_area = position_df['shot_zone_area'][0]
    shot_zone_basic = position_df['shot_zone_basic'][0]
    shot_zone_range = position_df['shot_zone_range'][0]
    opponent = teams_slider
    action= shot_est
    away= away_radio

    features = ['minutes_remaining','last_seconds','period', 'playoffs', 'season', 'shot_distance', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range','opponent', 'action','away']

    input_df.fillna(0, inplace=True)
    for f in features:
        input_df[f+'_'+str(eval(f))]=1

    input_df['minutes_remaining_0']=0    

    success_prob_df = pd.Series(xgb_clfCV.predict_proba(input_df)[:,1])

    st.write('XGBoost ouput:')
    shot_est2['shot_success_probability'] = success_prob_df
    st.write(shot_est2)

    if st.checkbox('Show court plot'):
        fig, ax = plt.subplots()
        plt.scatter(x=x_slider, y=y_slider)
        plt.ylabel('loc_y')
        plt.xlabel('loc_x')
        plt.xlim(-250, 250)
        plt.ylim(-50, 420)
        ax.set_aspect(500/470)
        draw_court(outer_lines=True)
        st.pyplot(fig)
