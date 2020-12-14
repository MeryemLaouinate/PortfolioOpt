import streamlit as st
from multiapp import MultiApp
from apps import ETF, bourCasa # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("S&P500 Stock Market", ETF.app)
app.add_app("Moroccan Stock Market", bourCasa.app)

# The main app
app.run()