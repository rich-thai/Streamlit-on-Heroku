import streamlit as st
import pandas as pd
# import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
# from nba_defs import draw_court
# import os
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




st.title('Hello World')
st.markdown('This is an analysis of all the shots Kobe Bryant has made throughout his career, including missed shots. This is identified with the shot_made_flag.')
st.markdown('**Data source**: https://www.kaggle.com/c/kobe-bryant-shot-selection/overview')
st.markdown('**Problem:** There are 5000 shot_made_flag values are missing, so the task is estimate the probability that the shot was a sucess.')
st.markdown('**Approach:** Clean the dataframe of redundant features, visualize the data for any outliers, engineer additional features that maybe useful, develop a binary classifier for the shot_made_flag probability.')

url = 'https://raw.githubusercontent.com/rich-thai/Streamlit-on-Heroku/master/data.csv'

# @st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(url, index_col=0).reset_index()
    return data

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data (first 100)'):
    st.subheader('Raw data (first 100)')
    st.write(df.head(100))

st.markdown('The shape of the dataframe is: '+str(df.shape))

st.subheader('Cleaning the data')
st.markdown('* Kobe only played for the Los Angeles Lakers for his entire career, so the features "team_id" and "team_name" are not needed.')
st.markdown('* The "matchup" feature contains "@" if it is an away game, and "vs" for a home game. Create an "away" binary feature.')
st.markdown('* The "matchup" feature is redundant with "opponent", so this can also be dropped.')
st.markdown('* The game date is not important for a professional athlete. His average performance over a season is more useful.')
st.markdown('* lat is linear with loc_y, and lon is linear with loc_x. Is is not necessary to keep both, so I will drop lat and lon.')
st.markdown('* Sort data by game_date and game_event_id.')
st.markdown('* Drop shot_id, game_id, and game_event_id.')



# cleaning data
df['game_date'] = pd.to_datetime(df['game_date'])
df.sort_values(by=['game_date','game_event_id'], inplace=True)
df['away'] = df['matchup'].str.contains('@')
df = df.drop(['team_id','team_name','matchup','game_date','lat','lon','shot_id','game_event_id', 'game_id'], axis=1)
df['period_minutes_remaining'] = df['minutes_remaining'] + df['seconds_remaining']/60


if st.checkbox('Show cleaned dataframe'):
    st.write(df.head())

seasons = df.season.unique().tolist()
    
start_season, end_season = st.select_slider(
     'Select a season',
     options=seasons,
     value=(seasons[0], seasons[-1]))
st.write('You selected seasons between', start_season, 'and', end_season)
index1 = seasons.index(start_season)
index2= seasons.index(end_season)

action_df = df[df['season'].isin(seasons[index1:index2])]['action_type']\
    .value_counts().reset_index()\
    .rename(columns={'index':'action type', 'action_type':'Shots Attempted'})

# st.write(alt.Chart(action_df).mark_bar().encode(
#     y=alt.X('action type', sort=None),
#     x='Shots Attempted',
# ))

#---------------------------------
st.subheader('Visualize Shot Selection')
st.markdown('Spatial Shot Distribution')
# create shot location scatter plot
shotloc_df = df[df['season'].isin(seasons[index1:index2])]
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



# create shot location scatter plot
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

#--------
st.subheader('Additional Cleaning')
st.markdown('Machine learning relies on recognizing patterns from large sets of data. There are some rare shots that can generalized.')
st.markdown('* There are many action_types that were not used often. Set a threshold of 20 shot attempts. An action_type below threshold is labelled as combined_shot_type. Drop the combined_shot_type column.')
st.markdown('* There is not much data for shot_distance>30ft. Set an upper limit at 35ft.')
st.markdown('* The specific location of loc_x and loc_y should not be critical, but the general locations from shot_zone_area, shot_zone_basic, shot_zone_range and shot_distance should be.')
st.markdown('* Kobe is trusted to take the last shot in each period. Perhaps the seconds_remaining feature is only useful when there is less than a minute left. Create a "last_seconds" binary feature with a 10s threshold.')

action_count = df['action_type'].value_counts()
df['action'] = df.apply(lambda x: x['action_type'] if action_count[x['action_type']]>20 else x['combined_shot_type'], axis=1)
df['shot_distance'] = df['shot_distance'].clip(0,35)
df['last_seconds'] = (df['period_minutes_remaining']<=10/60)
df.drop(['action_type','combined_shot_type','loc_x','loc_y','seconds_remaining','period_minutes_remaining'], axis=1, inplace=True)

st.write(df.head())



#-----------------------------------------------
st.subheader('Machine Learning')
st.markdown('Models are evaluated by log-loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))')
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

st.markdown('The data is separated into a training (X_train_full, y_train_full) and test set (X_test) based on if the shot_made_flag = null.')
st.markdown('Split the training data into 80% for training (X_train, y_train), 20% for validation (X_valid, y_valid).')
st.markdown('**Strategy:** Use grid search to find optimal hyperparameters and k-fold cross-validation.')
st.markdown('**Models:** Decision tree, Random forest, XGBoost.') 



# feature_names = df.drop('shot_made_flag', axis=1).columns.tolist()
# cat_attribs = df.select_dtypes(include=object).columns.tolist()
# num_attribs = df.select_dtypes(exclude=object).drop('shot_made_flag', axis=1).columns.tolist()

# #--------------------------
# st.subheader('Logistic Regression')

# num_pipeline = Pipeline([
#  ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
#     ,('std_scaler', StandardScaler())
# ])
# full_pipeline = ColumnTransformer([
# ("num", num_pipeline, num_attribs),
# ("cat", OrdinalEncoder(), cat_attribs),
# ])

# X_train_transformed = full_pipeline.fit_transform(X_train_full)
# X_test_transformed = full_pipeline.fit_transform(X_test)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


# logclf = LogisticRegression(random_state=0)
# logclf_state = st.text('Running logistic regression...')
# scores = cross_val_score(logclf, X_train_transformed, y_train_full,
# scoring=LogLoss, cv=10)
# logclf_state.text('Logistic regression complete.')
# st.write(str(scores.mean()) + '+-' + str(scores.std()))


# #-------------
# st.subheader('k-Nearest Neighbours Classifier')

# num_pipeline = Pipeline([
#  ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
#     ,('std_scaler', StandardScaler())
# ])
# full_pipeline = ColumnTransformer([
# ("num", num_pipeline, num_attribs),
# ("cat", OneHotEncoder(), cat_attribs),
# ])

# X_train_transformed = full_pipeline.fit_transform(X_train_full)
# X_test_transformed = full_pipeline.fit_transform(X_test)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


# knn = KNeighborsClassifier()
# knn_state = st.text('Running kNN classifier...')
# scores = cross_val_score(knn, X_train_transformed, y_train_full,
# scoring=LogLoss, cv=10)
# knn_state.text('kNN classifier complete.')
# st.write(str(scores.mean()) + '+-' + str(scores.std()))



#------------------
st.subheader('Tree-Based Methods')
st.markdown('* One of the most intuitive types of machine learning models.')
st.markdown('* Finds the best feature and threshold that minimizes the impurity down the tree (gini).')
st.markdown('* Can handle categorical features with ordinal encoding or one-hot encoding.')
st.markdown('* Does not require feature scaling as opposed to other types of models.')

st.markdown('With one-hot encoding and ensemble methods, we can determine which features are important after training the model.')



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




#------------------------------
st.subheader('Decision Tree Classifier')
st.markdown('The ')

param_grid = [
{'max_depth': [2,3,4,5], 'max_features':[2,4,8,16,None], 'max_leaf_nodes':[2,3,4,None]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
dtree = DecisionTreeClassifier(random_state=0)
dtree_state = st.text('Loading decision tree classifier...')
dtreeCV  = pickle.load(open('dtree_model.pkl', 'rb'))
dtree_state.text('Loaded decision tree classifier.')
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



#-----------------------------------------------
st.subheader('Random Forest Classifier')
st.markdown('An ensemble of decision trees where subsets of data is divided amongst tree. The ensemble votes for the most likely outcome.')

param_grid = [
{'max_depth': [7], 'max_features':[5,10,15]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
RFclf = RandomForestClassifier(random_state=0, n_estimators=100)
RFclfCV = pickle.load(open('rfclf_model.pkl', 'rb'))
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


#-----------------------------------------------
st.subheader('Gradient Boosting with XGBoost')
st.markdown('Decision trees are sequentially fit to the residuals, and outputs are summed.')

xgb_clfCV = pickle.load(open('xgbclf_model.pkl', 'rb'))
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

