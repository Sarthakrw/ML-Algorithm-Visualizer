import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, plt_gradients

from functionality import  model_output, cost_output, gradient_descent_output

x_train = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7])
y_train = np.array([21.0, 47.0, 27.0, 75.0, 30.0, 20.0, 88.0, 60.0, 81.0, 25.0])

w = 10
b = 20

m = x_train.shape[0]

st.title("Single Feature Linear Regression Algorithm Visualization")
st.write("By : Sarthak Rawat")

st.header("Input Data")
# function to get user input data from manual input
def get_manual_input():
    x_train = st.text_input("Enter x training data (comma separated)", value="2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7")
    y_train = st.text_input("Enter y training data (comma separated)", value="21.0, 47.0, 27.0, 75.0, 30.0, 20.0, 88.0, 60.0, 81.0, 25.0")
    # split the comma-separated string to get individual data points
    x_train = np.array([float(val.strip()) for val in x_train.split(",")])
    y_train = np.array([float(val.strip()) for val in y_train.split(",")])

    return x_train, y_train, len(x_train)

# function to get user input data from CSV upload
def get_random_input():
    m = st.number_input("Enter Number of Examples", value=2,step=1,min_value=2)
    x_train = np.sort(np.random.randint(0, 101, size=m))
    y_train = np.sort(np.random.randint(0, 101, size=m))
    
    return x_train, y_train, m

# set up sidebar for user input options
input_option = st.selectbox("Select Input Option", ["Random Input", "Manual Input"])

# get user input data based on selected option
if input_option == "Random Input":
    x_train, y_train, m = get_random_input()
elif input_option == "Manual Input":
    x_train, y_train, m = get_manual_input()

# display input data
if x_train is not None and y_train is not None:
    st.write("Training data:")
    df = pd.DataFrame({'x_train': x_train, 'y_train': y_train})
    st.dataframe(df, width=500)

else:
    st.warning("Please select an input option to proceed.")

st.header("Plotting Points")

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, marker='x', c='r')
st.pyplot(fig)

st.markdown(r"""
            ## Computing Model Function
            
            The hypothesis function $f_{w,b}(x^{(i)})$ for a given input feature $x^{(i)}$ is defined as:
            
            $$ f_{w,b}(x^{(i)}) = wx^{(i)} + b$$
            
            """)



f_wb = model_output(x_train, w, b, m)


f_wb_df = pd.DataFrame(f_wb, columns=['f_wb'])
st.dataframe(f_wb_df, width=500)



ax.plot(x_train, f_wb)
st.pyplot(fig)

st.header("Computing Cost")
st.latex(r"""
    \begin{aligned}
    \text{Cost Function} \\
    J(w,b) &= \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{2}
    \end{aligned}
""")


cost = cost_output(x_train, y_train, w, b, m)

st.write("Calculated Initial Cost: ", cost, unsafe_allow_html=True, 
         style="font-size: 50px; color: green")



st.markdown(r"""
            ## Gradient Descent Algorithm

            $$
            \begin{aligned}
            \text{repeat until convergence:} \; \lbrace \\
            \;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3} \; \\
             b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \\
            \rbrace
            \end{aligned}
            $$

            ## Gradients
    
            The gradient is defined as:
            $$
            \begin{align}
            \frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
            \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
            \end{align}
            $$


            """)

#initalizing parameters
iterations = 100000
alpha = 0.0001

w_init = 0
b_init = 0
#######################


w_final, b_final = gradient_descent_output(x_train, y_train, w_init, b_init, alpha, iterations, m)

st.header("Computed Optimized Value of w and b")

st.write("Calculated Value of w = ", + w_final)
st.write("Calculated Value of b = ", + b_final)

st.header("Final Cost after Fitting")
st.write("Final Cost = ", +cost_output(x_train, y_train, w_final, b_final, m))

st.header("Plotting Fitted graph")
fig3, ax3 = plt.subplots()

ax3.plot(x_train, model_output(x_train, w_final, b_final, m), c = 'b', label = 'predicted value')
ax3.scatter(x_train, y_train, marker = 'x', c = 'r', label = 'actual value')
plt.title("Fitted Graph")


plt.legend()
st.pyplot(fig3)

st.header("Make a Prediction")

user_input = st.number_input("Enter value : ")

st.write(f"Predicted Value : {(w_final*user_input) + b_final:.2f}")

st.header("")



