# CodeAlpha Data Science Internship Projects
    
     This repository contains the projects completed for the CodeAlpha data science internship. It        
      includes two main projects: a sales prediction model and an analysis of unemployment in India.       
    
     ---
    
   ## 1. Sales Prediction with a Web UI
    
     This project predicts product sales based on advertising spending on TV, Radio, and Newspaper.       
     It includes a trained machine learning model and a user-friendly web interface to make
      predictions.
   
   ### Files:
    - `Sales_Prediction/app.py`: The Flask web server that runs the application.
    - `Sales_Prediction/sales_prediction.py`: The script to train the sales prediction model.
    - `Sales_Prediction/sales_prediction_model.pkl`: The saved, pre-trained machine learning model.      
    - `Sales_Prediction/Advertising.csv`: The dataset used for training.
    - `Sales_Prediction/templates/index.html`: The HTML file for the web user interface.
   
   ### How to Run the Web App:
    1.  Make sure you have Python and the required libraries (`Flask`, `pandas`, `scikit-learn`)
      installed.
    2.  Navigate to the `Sales_Prediction` folder in your terminal.
    3.  Run the command: `python app.py`
    4.  Open your web browser and go to `http://127.0.0.1:5000` to use the application.
   
   ### Model Performance:
   The model is highly accurate and performs very well:
    - **R-squared:** 0.899 (This means the model explains about 89.9% of the variation in sales).        
    - **Mean Squared Error (MSE):** 3.17 (The average squared difference between the actual and
      predicted sales is very low).
   
    ---
   
   ## 2. Unemployment Analysis in India
  
    This project analyzes the unemployment rate in India using data up to November 2020. It includes     
      data cleaning, exploratory data analysis, and a predictive model for the unemployment rate.
   
   ### Files:
    - `Unemployment_Analysis/Unemployment_Analysis.py`: The script for data exploration and
      visualization.
    - `Unemployment_Analysis/unemployment_model.py`: The script to train the unemployment prediction     
      model.
    - `Unemployment_Analysis/unemployment_model.pkl`: The saved, pre-trained machine learning model.     
    - `Unemployment_Analysis/Unemployment_Rate_upto_11_2020.csv`: The primary dataset used for the       
      analysis.
    - `Unemployment_Analysis/unemployment_rate_by_state.png`: A plot showing the unemployment rate       
      by state.
    - `Unemployment_Analysis/unemployment_rate_over_time.png`: A plot showing the unemployment rate      
      over time.
   
   ### Model Performance:
    The linear regression model for unemployment prediction had a low R-squared value of 5.6%,
      indicating that the provided features are not sufficient to accurately predict the unemployment      
      rate. Further feature engineering would be needed to improve this model.
   
    ---
   
   ## Technologies Used
    - Python
    - Pandas
    - Scikit-learn
    - Flask
    - HTML/CSS
    - Matplotlib / Seaborn
