import streamlit as st
import pandas as pd
# import altair as alt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
# from nba_defs import draw_court
# import os

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

@st.cache(allow_output_mutation=True)
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
st.markdown('* The "matchup" feature is redundant with "opponent", so this can also be dropped.')
st.markdown('* The "game_date" feature can be split by year, month, day')
st.markdown('* lat is linear with loc_y, and lon is linear with loc_x. Is is not necessary to keep both, so I will drop lat and lon.')
st.markdown('* Sort data by game_date and game_event_id.')
st.markdown('* Drop shot_id and game_event_id.')

# cleaning data
df['game_date'] = pd.to_datetime(df['game_date'])
df.sort_values(by=['game_date','game_event_id'], inplace=True)
df['year'] = df['game_date'].dt.year
df['month'] = df['game_date'].dt.month
df['day'] = df['game_date'].dt.day
df = df.drop(['team_id','team_name','matchup','game_date','lat','lon','shot_id','game_event_id'], axis=1)
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
ax.legend([0,1,2])
draw_court(outer_lines=True)
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(7,10))
ax.barh(action_df['action type'].values, action_df['Shots Attempted'].values, align='center')
plt.xscale('log')
plt.xlabel('Shot Attempts')
st.pyplot(fig)


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


#-----------------------------------------------
st.subheader('Machine Learning')
st.markdown('Models are evaluated by log-loss: -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))')
LogLoss = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

st.markdown('The data is separated into a training and test set based on if the shot_made_flag = null.')
st.markdown('Split the training data into 80% for training, 20% for validation.')
st.markdown('**Strategy:** Use grid search to find optimal hyperparameters and k-fold cross-validation.')

## Split into train/test set based on missing value
X_test = df[df['shot_made_flag'].isnull()].drop('shot_made_flag',axis=1)
X_train_full = df[df['shot_made_flag'].notnull()]
y_train_full = X_train_full.pop('shot_made_flag')

feature_names = df.drop('shot_made_flag', axis=1).columns.tolist()
cat_attribs = df.select_dtypes(include=object).columns.tolist()
num_attribs = df.select_dtypes(exclude=object).drop('shot_made_flag', axis=1).columns.tolist()

#--------------------------
st.subheader('Logistic Regression')

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
    ,('std_scaler', StandardScaler())
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OrdinalEncoder(), cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


logclf = LogisticRegression(random_state=0)
logclf_state = st.text('Running logistic regression...')
scores = cross_val_score(logclf, X_train_transformed, y_train_full,
scoring=LogLoss, cv=10)
logclf_state.text('Logistic regression complete.')
st.write(str(scores.mean()) + '+-' + str(scores.std()))


#-------------
st.subheader('k-Nearest Neighbours Classifier')

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
    ,('std_scaler', StandardScaler())
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", OneHotEncoder(), cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)


knn = KNeighborsClassifier()
knn_state = st.text('Running kNN classifier...')
scores = cross_val_score(knn, X_train_transformed, y_train_full,
scoring=LogLoss, cv=10)
knn_state.text('kNN classifier complete.')
st.write(str(scores.mean()) + '+-' + str(scores.std()))



#------------------
st.subheader('Tree-Based Methods')
st.markdown('* Transform string features to ordinal encoding (0,1,2...).')
st.markdown('* Does not require feature scaling.')
st.markdown('* Finds the best feature and threshold that minimizes the impurity down the tree (gini).')



## Use an ordinal encoder to label the string features
encoder = OrdinalEncoder()

num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")) # even though there are no missing values
])
full_pipeline = ColumnTransformer([
("num", num_pipeline, num_attribs),
("cat", encoder, cat_attribs),
])

X_train_transformed = full_pipeline.fit_transform(X_train_full)
X_test_transformed = full_pipeline.fit_transform(X_test)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_transformed, y_train_full, test_size=0.2, random_state=42)





#------------------------------
st.subheader('Decision Tree Classifier')

param_grid = [
{'max_depth': [2,3,4,5], 'max_features':[2,4,8,16,None], 'max_leaf_nodes':[2,3,4,None]}
]
st.markdown('Grid Search Parameters with ' + str(param_grid))
dtree = DecisionTreeClassifier(random_state=0)
dtree_state = st.text('Running decision tree classifier...')
dtreeCV = GridSearchCV(dtree, param_grid, cv=10,
                       scoring=LogLoss,
                       return_train_score=True)
dtreeCV.fit(X_train_transformed, y_train_full)
dtree_state.text('Decision tree classifier complete.')
st.text(dtreeCV.best_params_)
st.text([dtreeCV.best_score_, dtreeCV.best_estimator_])


fig = plt.figure(figsize=(7,7))
_ = tree.plot_tree(dtreeCV.best_estimator_, 
                   feature_names=num_attribs+cat_attribs,  
                   class_names=['0','1'],
                   filled=True)
fig.canvas.draw()
figdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
figdata = figdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
st.image(figdata)


# #-----------------------------------------------
# st.subheader('Random Forest Classifier')
# st.markdown('An ensemble of decision trees where subsets of data is divided amongst tree. The ensemble votes for the most likely outcome.')

# param_grid = [
# {'max_depth': [2,4,8], 'max_features':[2,4,8,16]}
# ]
# st.markdown('Grid Search Parameters with ' + str(param_grid))
# RFclf = RandomForestClassifier(random_state=0, n_estimators=20)
# RFclfCV = GridSearchCV(RFclf, param_grid, cv=10,
#                        scoring=LogLoss,
#                        return_train_score=True)
# RFclfCV.fit(X_train_transformed, y_train_full)
# st.text(RFclfCV.best_params_)
# st.text([RFclfCV.best_score_, RFclfCV.best_estimator_])