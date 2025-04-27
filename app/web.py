
import streamlit as st
import pandas as pd
import time
import torch
from route_model import create_model
import numpy as np


@st.cache_data
def load_data():
    df1 = pd.read_parquet('data_part_1.parquet')
    df2 = pd.read_parquet('data_part_2.parquet')
    df3 = pd.read_parquet('data_part_3.parquet')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    return df


def load_model(model, path="1-model.pth"):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Introduction", "Database Query", "Model Prediction"))

################
# Introduction
################
if page == "Introduction":

    st.title("Welcome to Our Application: AI for Non-aqueous Electrolyte Design")
    # st.image("your_intro_image.png", use_column_width=True)
    st.write("""
    This platform allows you to query our non-aqueous electrolyte database and predict properties using our deep learning models.
    It accelerates the discovery of non-aqueous electrolyte for battery and energy storage applications.
    """)
    st.image("./web-1.png")
    st.subheader("Core functions: ")
    st.write("1. The detailed information for over 10 million non-aqueous electrolytes is provided. One can search the non-aqueous electrolyte using the inforamtion of Li-salts, solvents, and conditions.")
    st.write("2. An online predictive platform is provided. One can predict ionic conductivity for non-aqueous electrolyte by setting custom inputs.")
    st.write("3. (Underdeveloped) The development of a data sharing platform allows researchers to upload their own data, improve the scale and quality of data, and jointly build a data platform to promote the development of electrolyte materials.")
    st.write("4. (Underdeveloped) By establishing a cloud training platform, researchers can train small-scale models online and continuously improve the predictive capabilities of the models.")

    st.subheader("Highlights: ")
    st.write("1. A multi-feature fusion network is built for predicting the ionic conductivity of non-aqueous electrolytes.")
    st.write("2. A dynamic-routing strategy is incorporated for improving the model predictive performance, achiveing a benchmark error of 0.372 mS/cm, reducing error by 65.4% over existing models.")
    st.write("3. Conductivity atlas across 11,515,140 non-aqueous electrolytes are shaped, generating the largest virtual dataset to date.")
    st.write("4. LiFSI-, LiTFSI-, and LiBOB-based non-aqueous electrolytes have been validated to have ionic conductivity > 20 mS/cm.")
    st.write("5. This work also provides chemical insight into how molecular flexibility and ion-solvent interactions influence conductivity by incorporating the gradient-decoupling approach, symbolic regression, and quantum chemistry calculation.")

    st.subheader("Main developers: ")
    st.write("Professor Fengqi You (you.fengqi@gmail.com), Dr. Zhilong Wang (zhilongwang.ai@gmail.com)")
    st.write("Please let us know if you have any questions!")

    st.subheader("Copyright: ")
    st.write("""Process-Energy-Environmental Systems Engineering (PEESE) lab at Cornell University: https://www.peese.org/""")

    st.subheader("Acknowledgement: ")
    st.write(
        """This project is partially supported by the Eric and Wendy Schmidt AI in Science Postdoctoral Fellowship, a program of Schmidt Sciences, LLC.""")

    st.subheader("Citing: ")
    st.write("""Zhilong Wang, Fengqi You*, submitted (2025).""")


################
# Database Query
################
elif page == "Database Query":
    st.title("Database Query")
    st.write("""
            Here, we provide data on more than 10 million non-aqueous electrolytes, 
            which can be searched based on the combination of Li-salt, organic solvent, temperature and other information.
            """)

    st.subheader("Li-salt selection")
    Li_salt_options = ['LiPF6', 'LiBF4', 'LiTDI', 'LiFSI', 'LiTFSI', 'LiPDI', 'LiClO4', 'LiAsF6', 'LiBOB', 'LiCF3SO3', 'LiBPFPB', 'LiBMB', 'LiN(CF3SO2)2']
    Li_salt = st.selectbox("13 types of Li-salts are provided:", Li_salt_options)
    concentration = st.slider("The concentration of Li-salt: ", 0.2, 2.0, 0.6, step=0.2)
    concentration_unit_options = ['mol/l', 'mol/kg']
    unit = st.selectbox("Two types of Li-salt concentration units can be selected:", concentration_unit_options)



    st.subheader("Organic solvent selection")
    st.write("""Note:
        Currently, we only support the exploration of non-aqueous electrolyte systems composed of two organic solvents, the sum of their ratios should be 1.0.
        For example, EC: PC = 0.5: 0.5.
        """)
    solvent_1_options = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'MC', 'THF', 'Toluene', 'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-O', '3-MeSul', 'Ethyldg', 'DMF', 'Ethylb', 'Ethylmg', 'Benzene', 'g-Buty', 'Cumene', 'PropSul', 'Pseudo', 'TEOS', 'm-Xylene', 'o-Xylene']
    solvent_1 = st.selectbox("The first solvent (38 types of solvents are provided):", solvent_1_options)
    solvent_1_ratio = st.slider("The ratio of the first solvent molecule: ", 0.0, 0.9, 0.5, step=0.1)

    solvent_2_options = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'MC', 'THF', 'Toluene', 'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-O', '3-MeSul', 'Ethyldg', 'DMF', 'Ethylb', 'Ethylmg', 'Benzene', 'g-Buty', 'Cumene', 'PropSul', 'Pseudo', 'TEOS', 'm-Xylene', 'o-Xylene']
    solvent_2 = st.selectbox("The second solvent (38 types of solvents are provided):", solvent_2_options)
    solvent_2_ratio = st.slider("The ratio of the second solvent molecule: ", 0.0, 0.9, 0.5, step=0.1)

    ratio_unit_options = ['w', 'v', 'mol']
    solvent_unit = st.selectbox("Three types of ratio units (weight; volume; mol) can be selected:", ratio_unit_options)


    st.subheader("Temperature")
    temperature = st.slider("Temperature (from 200 K to 320 K, unit: 100 K): ", 2.0, 3.2, 3.0, step=0.2)

    st.write(
        "Note: since we have over 10 million entries, the data loading process may be a little slow, please be patient.")
    if st.button("**Submit**"):
        data = load_data()
        filtered_data = data[(data['Li-salt'] == Li_salt) &
                             (data[solvent_1] == solvent_1_ratio) &
                             (data[solvent_2] == solvent_2_ratio) &
                             (data['T'] == temperature) &
                             (data['concentration'] == concentration) &
                             (data['concentration-unit'] == unit) &
                             (data['solvent-unit'] == solvent_unit)]
        if filtered_data.empty:
            st.warning("No data found for the selected combination.")
        else:
            st.success(f"Found {len(filtered_data)} entries.")
            pages = len(filtered_data) // 100 + 1
            page_number = st.number_input("Page number:", min_value=1, max_value=pages, value=1)
            start_idx = (page_number - 1) * 100
            end_idx = start_idx + 100
            st.dataframe(filtered_data.iloc[start_idx:end_idx])



##################
# Model Prediction
##################
elif page == "Model Prediction":
    salt_properties = {

        'LiPF6': [0.22278622, 0.0498452, 0, 0, 0, 0, 0, 0.19672131, 0.22278622, 0.19047619, 0, 0, 0.19227575,
                  0.14135673],
        'LiBF4': [0.13748968, -0.21878225, 0, 0, 0, 0, 0, 0.13114754, 0.13748968, 0.14285714, 0, 0, 0.13501987,
                  0.11541728],
        'LiFSI': [0.27436924, -0.41373839, 0.78307985, 1, 0, 0, 0, 0.2295082, 0.27436924, 0.23809524, 0, 0.5,
                  0.23124724, 0.2139321],
        'LiTDI': [0.28163641, -0.28315789, 0.7088403, 0, 1, 0, 0, 0.2704918, 0.28163641, 0.33333333, 0, 0.375,
                  0.30185877, 0.3454273],
        'LiTFSI': [0.42105325, 0.00917183, 0.67984791, 1, 0, 0, 0, 0.37704918, 0.42105325, 0.38095238, 0, 0.5,
                   0.40738047, 0.38403373],
        'LiClO4': [0.15603528, -1, 0.87680608, 0, 0, 0, 0, 0.13114754, 0.15603528, 0.14285714, 0, 0.5, 0.13501987,
                   0.11541728],
        'LiAsF6': [0.28724189, -0.11037152, 0, 0, 0, 0, 0, 0.19672131, 0.28724189, 0.19047619, 0, 0, 0.19227575,
                   0.14135673],
        'LiBMB': [0.32536266, -0.59259546, 1, 0, 0, 0, 0, 0.32786885, 0.31944919, 0.38095238, 0, 1, 0.33161006,
                  0.40336038],
        'LiBOB': [0.28421769, -0.69324045, 1, 0, 0, 0, 0, 0.27868852, 0.28421769, 0.33333333, 0, 1, 0.28917741,
                  0.34759499],
        'LiBPFPB': [1, 0.5125387, 0.35095057, 0, 0, 0, 0, 1, 1, 1, 0, 0.5, 1, 1],
        'LiCF3SO3': [0.22880969, -0.37985036, 0.54372624, 0, 0, 0, 0, 0.20491803, 0.22880969, 0.21428571, 0, 0.375,
                     0.21003091, 0.18755309],
        'LiPDI': [0.34627687, 0.23312178, 0.72490494, 0.5, 1, 1, 0, 0.3442623, 0.34479851, 0.38095238, 1, 0.375,
                  0.37686981, 0.4175631],
        'LiN(CF3SO2)2': [0.42105325, -0.2498323, 0.78307985, 1, 0, 0, 0, 0.37704918, 0.42105325, 0.38095238, 0, 0.5,
                         0.38126932, 0.3582037]
    }

    solvent_properties = {
        'EC': [0.25594508, 0.03998747, 0.76988082, 0, 0, 0, 0, 0.27868852, 0.24859623, 0.3, 0, 0.6, 0.27124761,
               0.33566168],
        'PC': [0.29671342, 0.14139173, 0.76988082, 0, 0, 0, 0, 0.32786885, 0.28412984, 0.35, 0, 0.6, 0.32482532,
               0.38134457],
        'DMC': [0.26180442, 0.10419712, 0.76988082, 0, 0, 0, 0, 0.29508197, 0.24859623, 0.3, 0, 0.6, 0.30731233,
                0.32571118],
        'EMC': [0.30257276, 0.206019, 0.76988082, 0.08333333, 0, 0, 0, 0.3442623, 0.28412984, 0.35, 0, 0.6, 0.35084631,
                0.38370694],
        'DEC': [0.3433411, 0.30784089, 0.76988082, 0.16666667, 0, 0, 0, 0.39344262, 0.31966345, 0.4, 0, 0.6, 0.39438029,
                0.44170269],
        'DME': [0.2619323, 0.07287534, 0.4, 0.25, 0, 0, 0, 0.31147541, 0.23679804, 0.3, 0, 0.4, 0.2972686, 0.33802404],
        'DMSO': [0.22709596, -0.00138338, 0.36988082, 0, 0, 0, 0, 0.21311475, 0.21326675, 0.2, 0, 0.2, 0.22024438,
                 0.2009032],
        'AN': [0.11931722, 0.13830654, 0.51549296, 0, 0, 0, 0, 0.13114754, 0.11250584, 0.15, 0, 0.2, 0.16666667,
               0.16403677],
        'MOEMC': [0.3898409, 0.10852996, 0.96988082, 0.25, 0, 0, 0, 0.44262295, 0.36699525, 0.45, 0, 0.8, 0.43791427,
                  0.49969845],
        'TFP': [1, 1, 0.96988082, 0.5, 0, 0, 0, 1, 1, 1, 0, 0.8, 1, 1],
        'EA': [0.25607296, 0.14862184, 0.56988082, 0.08333333, 0, 0, 0, 0.29508197, 0.23679804, 0.3, 0, 0.4, 0.30731233,
               0.32130294],
        'MA': [0.21530462, 0.04679996, 0.56988082, 0, 0, 0, 0, 0.24590164, 0.20126443, 0.25, 0, 0.4, 0.26377836,
               0.26330719],
        'FEC': [0.30823156, 0.11714345, 0.76988082, 0, 0, 0, 0, 0.32786885, 0.30480034, 0.35, 0, 0.6, 0.32482532,
                0.38134457],
        'DOL': [0.21530462, -0.00242744, 0.4, 0, 0, 0, 0, 0.24590164, 0.20126443, 0.25, 0, 0.4, 0.2176699, 0.28997879],
        '2-MeTHF': [0.2503415, 0.30938087, 0.2, 0, 0, 0, 0, 0.29508197, 0.22499985, 0.3, 0, 0.2, 0.27124761,
                    0.33566168],
        'DMM': [0.47150547, 0.2800167, 0.6, 0.5, 0, 0, 0, 0.55737705, 0.42626428, 0.55, 0, 0.6, 0.53502596, 0.6033771],
        'Freon 11': [0.39924898, 0.59605346, 0, 0, 0, 0, 0, 0.26229508, 0.40639256, 0.25, 0, 0, 0.27704855, 0.23198303],
        'Methylene chloride': [0.24685089, 0.37103257, 0, 0, 0, 0, 0, 0.16393443, 0.2453035, 0.15, 0, 0, 0.16666667,
                               0.16403677],
        'THF': [0.20957316, 0.20797661, 0.2, 0, 0, 0, 0, 0.24590164, 0.18946624, 0.25, 0, 0.2, 0.2176699, 0.28997879],
        'Toluene': [0.26780036, 0.5207298, 0, 0, 1, 0, 0, 0.29508197, 0.24873527, 0.35, 0, 0, 0.31478158, 0.39365743],
        'Sulfolane': [0.3492731, 0.05089789, 0.73976165, 0, 0, 0, 0, 0.3442623, 0.33166577, 0.35, 0, 0.4, 0.32805178,
                      0.37199717],
        '2-Glyme': [0.38996879, 0.07720819, 0.6, 0.5, 0, 0, 0, 0.45901639, 0.35519706, 0.45, 0, 0.6, 0.42787054,
                    0.51201131],
        '3-Glyme': [0.51800527, 0.08154103, 0.8, 0.75, 0, 0, 0, 0.60655738, 0.47359608, 0.6, 0, 0.8, 0.55847248,
                    0.68599859],
        '4-Glyme': [0.64604175, 0.08587388, 1, 1, 0, 0, 0, 0.75409836, 0.5919951, 0.75, 0, 1, 0.68907442, 0.85998586],
        '3-Me-2-O': [0.2938535, 0.01785341, 0.64008667, 0, 0, 0, 0, 0.32786885, 0.27823666, 0.35, 0, 0.4,
                                 0.32482532, 0.38329749],
        '3-MeSul': [0.39004145, 0.11510754, 0.73976165, 0, 0, 0, 0, 0.39344262, 0.36719938, 0.4, 0, 0.4,
                          0.38162949, 0.41768006],
        'Ethyldg': [0.38996879, 0.00830027, 0.8383532, 0.5, 0, 1, 0, 0.45901639, 0.35519706, 0.45, 1, 0.6,
                         0.42787054, 0.51201131],
        'DMF': [0.21244471, -0.07715598, 0.44008667, 0.08333333, 0, 0, 0, 0.24590164, 0.19537125, 0.25, 0, 0.2,
                0.26377836, 0.26330719],
        'Ethylb': [0.3085687, 0.58702234, 0, 0.08333333, 1, 0, 0, 0.3442623, 0.28426889, 0.4, 0, 0, 0.35831556,
                         0.45606143],
        'Ethylmg': [0.22116396, -0.09785446, 0.6383532, 0.16666667, 0, 1, 0, 0.26229508, 0.20126443, 0.25, 1,
                           0.4, 0.25373462, 0.28002829],
        'Benzene': [0.22703202, 0.4402276, 0, 0, 1, 0, 0, 0.24590164, 0.21320166, 0.3, 0, 0, 0.26120387, 0.34797454],
        'g-Buty': [0.25021362, 0.08441219, 0.56988082, 0, 0, 0, 0, 0.27868852, 0.23679804, 0.3, 0, 0.4,
                            0.27124761, 0.33566168],
        'Cumene': [0.34933705, 0.73345166, 0, 0.08333333, 1, 0, 0, 0.39344262, 0.3198025, 0.45, 0, 0, 0.41189327,
                   0.499289],
        'Propylsulfone': [0.43666913, 0.31875131, 0.73976165, 0.33333333, 0, 0, 0, 0.45901639, 0.40273299, 0.45, 0, 0.4,
                          0.45118446, 0.47803819],
        'Pseudocumeme': [0.34933705, 0.68173418, 0, 0, 1, 0, 0, 0.39344262, 0.3198025, 0.45, 0, 0, 0.421937,
                         0.48697614],
        'TEOS': [0.60549429, 0.40927125, 0.8, 0.66666667, 0, 0, 0, 0.6557377, 0.55668633, 0.65, 0, 0.8, 0.62532038,
                 0.72409335],
        'm-Xylene': [0.3085687, 0.60123199, 0, 0, 1, 0, 0, 0.3442623, 0.28426889, 0.4, 0, 0, 0.36835929, 0.43934033],
        'o-Xylene': [0.3085687, 0.60123199, 0, 0, 1, 0, 0, 0.3442623, 0.28426889, 0.4, 0, 0, 0.36835929, 0.44129325]
    }

    c_unit_encode = {
        'mol/kg': [0],
        'mol/l': [1]
    }

    solvent_ratio_type_code = {
        'mol': [1, 0, 0],
        'w': [0, 1, 0],
        'v': [0, 0, 1]
    }

    salt_features = []
    solvent_features = []
    condition_features = []


    model = load_model(create_model(), path="1-model.pth")
    st.title("Model Prediction")
    st.write("""
                Here, we provide an online function to predict conductivity for non-aqueous electrolytes.
                """)

    st.subheader("Li-salt selection")
    Li_salt_options = ['LiPF6', 'LiBF4', 'LiTDI', 'LiFSI', 'LiTFSI', 'LiPDI', 'LiClO4', 'LiAsF6', 'LiBOB', 'LiCF3SO3', 'LiBPFPB', 'LiBMB', 'LiN(CF3SO2)2']
    Li_salt = st.selectbox("13 types of Li-salts are provided:", Li_salt_options)
    concentration = st.slider("The concentration of Li-salt: ", 0.2, 2.0, 0.6, step=0.2)
    concentration_unit_options = ['mol/l', 'mol/kg']
    unit = st.selectbox("Two types of Li-salt concentration units can be selected:", concentration_unit_options)



    st.subheader("Organic solvent selection")
    st.write("""Note:
        Currently, we only support the exploration of non-aqueous electrolyte systems composed of two organic solvents, the sum of their ratios should be 1.0.
        For example, EC: PC = 0.5: 0.5.
        """)
    solvent_1_options = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'MC', 'THF', 'Toluene', 'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-O', '3-MeSul', 'Ethyldg', 'DMF', 'Ethylb', 'Ethylmg', 'Benzene', 'g-Buty', 'Cumene', 'PropSul', 'Pseudo', 'TEOS', 'm-Xylene', 'o-Xylene']
    solvent_1 = st.selectbox("The first solvent (38 types of solvents are provided):", solvent_1_options)
    solvent_1_ratio = st.slider("The ratio of the first solvent molecule: ", 0.0, 0.9, 0.5, step=0.1)

    solvent_2_options = ['EC', 'PC', 'DMC', 'EMC', 'DEC', 'DME', 'DMSO', 'AN', 'MOEMC', 'TFP', 'EA', 'MA', 'FEC', 'DOL', '2-MeTHF', 'DMM', 'Freon 11', 'MC', 'THF', 'Toluene', 'Sulfolane', '2-Glyme', '3-Glyme', '4-Glyme', '3-Me-2-O', '3-MeSul', 'Ethyldg', 'DMF', 'Ethylb', 'Ethylmg', 'Benzene', 'g-Buty', 'Cumene', 'PropSul', 'Pseudo', 'TEOS', 'm-Xylene', 'o-Xylene']
    solvent_2 = st.selectbox("The second solvent (38 types of solvents are provided):", solvent_2_options)
    solvent_2_ratio = st.slider("The ratio of the second solvent molecule: ", 0.0, 0.9, 0.5, step=0.1)

    ratio_unit_options = ['w', 'v', 'mol']
    solvent_unit = st.selectbox("Three types of ratio units (weight; volume; mol) can be selected:", ratio_unit_options)


    st.subheader("Temperature")
    temperature = st.slider("Temperature (from 200 K to 320 K, unit: 100 K): ", 2.0, 3.2, 3.0, step=0.2)

    salt_vector = salt_properties[Li_salt]
    salt_features.append(salt_vector)

    solvent_vector_1 = [i * solvent_1_ratio for i in solvent_properties[solvent_1]]
    solvent_vector_2 = [i * solvent_2_ratio for i in solvent_properties[solvent_2]]
    solvent_vector = [solvent_vector_1[i] + solvent_vector_2[i] for i in range(14)]
    solvent_features.append(solvent_vector)

    condition_vector = []
    condition_vector.append(temperature)
    condition_vector += c_unit_encode[unit]
    condition_vector += solvent_ratio_type_code[solvent_unit]
    condition_vector.append(concentration)
    condition_features.append(condition_vector)

    salt_features = torch.from_numpy(np.array(salt_features)).float()
    solvent_features = torch.from_numpy(np.array(solvent_features)).float()
    condition_features = torch.from_numpy(np.array(condition_features)).float()

    if st.button("**Predict**"):
        with torch.no_grad():
            predictions = model(salt_features, solvent_features, condition_features)
        with st.spinner("Loading the model, wait a moment..."):
            time.sleep(1)
            st.balloons()
        st.write(f"**Successful! the ionic conductivity at {temperature*100} K is predicted to be {predictions.item()} mS/cm. Thanks!**")


