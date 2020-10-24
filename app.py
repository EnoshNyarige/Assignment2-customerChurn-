import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model','rb'))


def predict_forest(contract_One, tenure, MonthlyCharges, payment_Electronic,TotalCharges):
    input=np.array([[contract_One, tenure, MonthlyCharges, payment_Electronic,TotalCharges]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("Enosh Nyarige")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Telco Customer Churn Prediction ML</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    contract_one_year = st.text_input("Contract one year? [0-No, 1-yes]","")
    tenure = st.text_input("Tenure [Total Period the customer has been getting service]","")
    monthlyCharges = st.text_input("Montly Charges [Monthly charges]","")
    payment_Electronic = st.text_input("Payment Electronic? [0-No, 1-yes]","")
    totalCharges = st.text_input("Total Charges [Total charges made]","")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Chances of churning are low</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Chances of churning are high</h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_forest(contract_one_year,tenure,monthlyCharges, payment_Electronic, totalCharges )
        st.success('The probability of churning is {}'.format(output))

        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()