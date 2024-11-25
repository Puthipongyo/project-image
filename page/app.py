# from page import Home, About, Upload, Visual
import streamlit as st
import pandas as pd
import numpy as np
import streamlit_option_menu
from streamlit_option_menu import option_menu
# from page.Home import show_home
# from page.About import show_about
# from page.Upload import show_upload
# from page.Visual import show_visual
from Home import show_home
from About import show_about
from Upload import show_upload
from Visual import show_visual



with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Home', 'Visual', 'Upload', 'About us'],
        icons = ['house', 'eye', 'upload', 'info-circle'],
        menu_icon = 'cast',
        default_index = 0
    )

def switch_case(option):
    if option == 'Home':
        return show_home()
    elif option == 'Visual':
        return show_visual()
    elif option == 'Upload':
        return show_upload()
    else : return show_about()

def main():
    switch_case(selected)

if __name__=='__main__':
    main()
