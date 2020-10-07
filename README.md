# Streamlit-on-Heroku

This is my first project using Streamlit to make a web application that is deployed on Heroku. The app can be found here: https://streamlit-testing.herokuapp.com/

I chose to look at the Kobe Bryant Shot Selection dataset once again. I demonstrate the data cleaning process and provide some visualizations that help to come up with new features.

The dataset is missing the outcome of 5000 shots, and the task is to implement a model to predict the outcome. I studied tree-based methods, using a decision tree classifier, random forest classifier and a gradient-boosted decision tree classifier (XGBoost).

I also trained a neural network to determine the different zones on the court and the most probable shot Kobe would have taken, and this was just based on X and Y locations.

Combining the XGBoost model for shot-success estimation and the neural network for shot-type probability, I created an interactive map where the X-Y location + additional features can be chosen. This will output the top 5 shots Kobe would have taken, and an estimate of the success probability.
