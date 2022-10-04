# imports
import streamlit as st
import streamlit.components.v1 as components
import pydeck as pdk
import numpy as np
import pandas as pd
import plotly.express as px
import scipy.sparse as ss
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances

alt.renderers.set_embed_options(theme='dark')


# Before we load the data, we determine the page configurations.
# page_title and page_icon determines the look of the browser tab.
# layout determines the overall layout of the page, i.e. wide means using the whole screen for inputs
st.set_page_config(
    page_title="JobHunter",
    page_icon="🔍",
    layout="wide",
)


# Using experimentatl_singleton and a definition, we load the data only once in order to minimise processing time
# this process is repeated for both the jobs and user_view dataset
# all lines within the definition is copy/pasted from the notebook
@st.experimental_singleton
def load_data_jobs():
    jobs = pd.read_csv('https://raw.githubusercontent.com/fr3drikk/the_dream_team-/main/App/Data/jobs.csv')

    # preprocess the data as in the notebook
    jobs = jobs.drop(['Industry', 'Salary', 'Address', 'Requirements'], axis=1)
    jobs = jobs.dropna()
    jobs['Listing.Start'] = pd.to_datetime(jobs['Listing.Start'])
    jobs['Listing.End'] = pd.to_datetime(jobs['Listing.End'])
    jobs['Created.At'] = pd.to_datetime(jobs['Created.At'])
    jobs['Updated.At'] = pd.to_datetime(jobs['Updated.At'])
    jobs['Employment.Type'] = jobs['Employment.Type'].replace(['Seasonal/Temp'], 'Temporary/seasonal')
    jobs['Employment.Type'] = jobs['Employment.Type'].fillna('Unspecified')
    jobs['Education.Required'] = jobs['Education.Required'].fillna('Unspecified')
    jobs['Education.Required'] = jobs['Education.Required'].replace(['Not Specified'] , 'Unspecified')

    # gruping positions into categories
    # Customer service
    jobs['Position'] = jobs['Position'].replace(['Customer Service Representative'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Customer Service / Sales ( New Grads Welcome!)'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Customer Service / Sales ( New Grads Welcome! )'] , 'Customer Service')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Sales / Customer Service – Part time / Full Time'] , 'Customer Service')

    # Accounting
    jobs['Position'] = jobs['Position'].replace(['Accounts Payable Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Accounting Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Accounts Receivable Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Bookkeeper'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Full Charge Bookkeeper'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Payroll Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Billing Clerk'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Payroll Administrator'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Staff Accountant'] , 'Accounting')
    jobs['Position'] = jobs['Position'].replace(['Senior Accountant'] , 'Accounting')

    # Sales
    jobs['Position'] = jobs['Position'].replace(['Sales Representative / Sales Associate ( Entry Level )'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Sales Associate'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Seasonal Wedding Sales Stylist'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate - Part-Time'] , 'Sales')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate'] , 'Sales')

    # Administration
    jobs['Position'] = jobs['Position'].replace(['Administrative Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Receptionist'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Front Desk Coordinator'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Executive Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['General Office Clerk '] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Office Assistant'] , 'Administration')
    jobs['Position'] = jobs['Position'].replace(['Medical Receptionist'] , 'Administration')

    # Restaurant
    jobs['Position'] = jobs['Position'].replace(['Bartender'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Server'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Kitchen Staff'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Hiring All Restaurant Positions - Servers - Cooks - Bartenders'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Cook'] , 'Restaurant personnel')
    jobs['Position'] = jobs['Position'].replace(['Kitchen Staff'] , 'Restaurant personnel')

    # Caregiving
    jobs['Position'] = jobs['Position'].replace(['Caregiver / Home Health Aide / CNA'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Registered Nurse'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Home Health Aide'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Certified Nursing Assistant'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Caregiver / Home Health Aide'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Caregiving'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Caregiver'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Entry Level Healthcare Professionals wanted for Caregiver Opportunities'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Registered Nurse (RN) / Licensed Practical Nurse (LPN) - Healthcare Nursing Staff'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Licensed Practical Nurse - LPN'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Physical Therapist'] , 'Caregiving professional')
    jobs['Position'] = jobs['Position'].replace(['Certified Nursing Assistant (CNA) - Healthcare Nursing Staff'] , 'Caregiving professional')

    # Human resources
    jobs['Position'] = jobs['Position'].replace(['Human Resources Assistant'] , 'Human resources')
    jobs['Position'] = jobs['Position'].replace(['Human Resources Recruiter'] , 'Human resources')
    jobs['Position'] = jobs['Position'].replace([''] , 'Human resources')

    # Retail professional
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate – Part-Time'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate / Photographer'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Retail Sales Associate - Part Time'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Part Time Retail Merchandiser'] , 'Retail professional')
    jobs['Position'] = jobs['Position'].replace(['Sales Representative - Retail'] , 'Retail professional')

    # Teacher 
    jobs['Position'] = jobs['Position'].replace(['Assistant Teacher'] , 'Teacher employee')
    jobs['Position'] = jobs['Position'].replace(['Teacher'] , 'Teacher employee')
    jobs['Position'] = jobs['Position'].replace([''] , 'Teacher employee')

    # Security officer
    jobs['Position'] = jobs['Position'].replace(['Security Officer'] , 'Security officer')
    jobs['Position'] = jobs['Position'].replace(['Security Officer - Regular'] , 'Security officer')
    jobs['Position'] = jobs['Position'].replace([''] , 'Security officer')
    jobs['Position'] = jobs['Position'].replace([''] , 'Security officer')
    jobs['Position'] = jobs['Position'].replace([''] , 'Security officer')

    # Driver professional
    jobs['Position'] = jobs['Position'].replace(['Part Time School Bus Drivers WANTED - Training Available'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['Delivery Driver (Part -Time)'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['School Bus Driver'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace(['Driver'] , 'Driver professional')
    jobs['Position'] = jobs['Position'].replace([''] , 'Driver professional')

    jobs = jobs[jobs['Position'].isin(['Customer Service', 'Accounting','Sales', 'Administration', 'Restaurant personnel', 'Caregiving professional', 'Human resources', 'Retail professional', 'Teacher employee','Security officer', 'Driver professional'])]

    return jobs

# repeating the process for user_view dataset
# Using experimentatl_singleton and a definition, we load the data only once in order to minimise processing time
@st.experimental_singleton
def load_data_user_view():
    user_view = pd.read_csv('https://raw.githubusercontent.com/fr3drikk/the_dream_team-/main/App/Data/user_job_views.csv')
    user_view = user_view.drop(['Industry'], axis=1)
    user_view['Company'] = user_view['Company'].fillna('Unspecified')
    user_view  = user_view.dropna(subset=['State.Name'])
    user_view['View.Duration'] = user_view['View.Duration'].fillna(user_view['View.Duration'].mean())
    user_view['Created.At'] = pd.to_datetime(user_view['Created.At'])
    user_view['Updated.At'] = pd.to_datetime(user_view['Updated.At'])
    user_view['View.Start'] = pd.to_datetime(user_view['View.Start'])
    user_view['View.End'] = pd.to_datetime(user_view['View.End'])

    # encode ids
    le_applicant = LabelEncoder()
    le_title = LabelEncoder()
    user_view['applicant_id'] = le_applicant.fit_transform(user_view['Applicant.ID'])
    user_view['title_id'] = le_title.fit_transform(user_view['Title'])

    # construct matrix
    ones = np.ones(len(user_view), np.uint32)
    matrix = ss.coo_matrix((ones, (user_view['applicant_id'], user_view['title_id'])))

    # decomposition
    svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    matrix_applicant = svd.fit_transform(matrix)
    matrix_title = svd.fit_transform(matrix.T)

    # distance-matrix
    cosine_distance_matrix_title = cosine_distances(matrix_title)

    return user_view, le_applicant, le_title, matrix, svd, matrix_applicant, matrix_title, cosine_distance_matrix_title



# Now we load the datasets before implementing it in streamlit
jobs = load_data_jobs()
user_view, le_applicant, le_title, matrix, svd, matrix_applicant, matrix_title, cosine_distance_matrix_title = load_data_user_view()



# defining figures, visualizations and calculations from the notebook


# defining the map
layer = pdk.Layer(
        "ScatterplotLayer",
        data=jobs[['Company','Position','State.Name', 'City', 'Employment.Type', "Longitude", "Latitude"]],
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=10,
        radius_min_pixels=1,
        radius_max_pixels=100,
        line_width_min_pixels=1,
        get_position=["Longitude", "Latitude"],
        get_radius="exits_radius",
        get_color=[255, 140, 0],
        get_line_color=[0, 0, 0],
    )

# Set the viewport location
view_state = pdk.ViewState(latitude=jobs['Latitude'].mean(), longitude=jobs['Longitude'].mean(), zoom=4, pitch=50)

# Renders
jobs_map = pdk.Deck(layers=[layer], 
initial_view_state=view_state,

#map_style='mapbox://styles/mapbox/light-v9',
tooltip={"text": "Company: {Company}\nPosition: {Position}\n Employment type: {Employment.Type}"})


# calculating various metrics
open_positions = jobs['Job.ID'].nunique()
companies_hiring = jobs['Company'].nunique()
active_jobhunters = user_view['Applicant.ID'].nunique()
avg_job_posts_viewed_per_jobhunter = (user_view['Job.ID'].value_counts().sum() / user_view['Applicant.ID'].nunique()).round(2)
avg_viewtime_per_job_post = (user_view['View.Duration'].mean() / 60).round(2)


# creating a pivot table in order to visualize a timeline dividable by employment type
jobs['Date'] = pd.to_datetime(jobs['Created.At']).dt.date #this could be moved to singleton
jobs_pivot = pd.pivot_table(jobs, values='Job.ID', index='Date', columns='Employment.Type', aggfunc='count')
positions_pivot = pd.pivot_table(jobs, values='Job.ID', index='Date', columns='Position', aggfunc='count')

# calculating the top 5 companies and positions posted
top_5_companies = jobs['Company'].value_counts().nlargest(5)
top_5_positions = jobs['Position'].value_counts().nlargest(5)

def similar_title(title, n):
  """
  this function performs city similarity search
  place: name of place (str)
  n: number of similar cities to print
  """
  ix = le_title.transform([title])[0]
  sim_title = le_title.inverse_transform(np.argsort(cosine_distance_matrix_title[ix,:])[:n+1])
  return sim_title[1:]

recommender_selectbox = pd.DataFrame(["Server @ Haven", "Server @ Oola Restaurant & Bar", "Server @ Burma Superstar", 
                                      "Server @ The Liberties Bar & Restaurant", "Server @ Sanraku Metreon", "Server @ COCO5OO", 
                                      "Server @ A La Turca", "Server @ The Liberty Cafe", "Server @ Yemeni's Restaurant", "Server @ L'Olivier", 
                                      "Waitstaff / Server @ Atria Senior Living", "Part Time Showroom Sales / Cashier @ Grizzly Industrial Inc.", 
                                      "Receptionist @ confidential", "Coordinator/Scheduler - IT @ Integrated Systems Analysts, Inc.", 
                                      "COMMUNITY ASSISTANT", "Part Time Errand/Clerical Assistant", "PART-TIME Administrative Assistant", 
                                      "Package Handler - Part-Time @ UPS", "Temporary Drivers @ Kelly Services", 
                                      "Customer Service Representative-Moonlighter @ U-Haul", "Pick-up Associate @ Orchard Supply Hardware", 
                                      "Part Time Liaison/Courier @ CIBTvisas.com", "NABISCO Part Time Merchandiser- Tucson 311 @ Mondelez International-Sales", 
                                      "Full Charge Bookkeeper Needed! @ Accountemps", "Entry Level Financial Analyst-Strong Excel Needed-Project! @ Accountemps",
                                      "Accountant @ Accountemps", "Accounting Manager / Supervisor @ Accountemps", "Accounts Payable Supervisor/Manager @ Accountemps",
                                      "Part Time Bookkeeper @ Accountemps", "Part Time Administrative Position in Omaha! @ Kelly Services", 
                                      "Mail Room Clerk @ OfficeTeam", "General Office Clerk @ OfficeTeam Healthcare", "DELIVERY DRIVERS @ Round Table Pizza",
                                      "Part-time School Bus Driver @ FirstGroup America", "92G Food Service Specialist @ Army National Guard", "School Bus Driver @ First Student", 
                                      "Business Consultants / Account Executives / Sales / (Inc.500/5000 Company) @ Central Payment", "Database Developer @ Spherion Staffing Services", 
                                      "Jr. Administrative Assistant @ OfficeTeam", 
                                      "Staff Nurse III @ University Health System", "Respiratory Therapist I @ University Health System", 
                                      "Administrative Assistant", "Part Time / Administrative/General Office - Part Time Administrative Assistant @ JobGiraffe",
                                      "Administrative Assistant - PT @ FCX Performance", "Marketing Assistant Human Resources @ MR-MRI St. Charles", 
                                      ])


# Streamlit deployment

# define a title that appears at the top of the page.
st.write('# Welcome to JobHunter 🔍')


# create tabs to browse through the app
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['Welcome', 'Find jobs by location', 'Find similar jobs', 'Find jobs by individuals', 'Facts about the job market', 'Tips and tricks for JobHunters'])



# within tab 1, an input line is created where users can write their name. The name is then included in a text that introduces the app to the user.
with tab1:
    st.markdown('#') # used to create spacing between the tabs and the name input
    name = st.text_input('Please enter your name and hit submit to get started on JobHunter!')

# once the user hits submit, the text below appears on the screen.
    if st.button('Submit'):
        st.write(' ')
        st.write('### Hi', name, '! 👋')
        st.write(' ')
        st.markdown(
            """
            Welcome to JobHunter, AAU's best tailored job searching platform for young professionals!

            - Are you looking for jobs in a specific area? Then browse through the map to find open positions in areas of your choice! 🌎 
            - Have you found an interesting job but wish to see similar opportunities? Then check out our type based job recommender! 📊 
            - Are you curious which jobs people similar to you are looking at? Then look up your applicant ID and explore our individual job recommender! 👤
            - Do you want to learn more about the job market in general? Then head to the fact page to spot numbers and trends! 📈
            - Have you found your dream job and are you ready to apply? Then make sure to use our tips and tricks to write a powerful CV and ace the interview! 📝
            """
        )



# within tab2, three multiselect boxes are created so the user can interact with the map. Filters do not work yet
with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        subset_data = jobs
        select_state = st.multiselect(
        'Select state',
        jobs.groupby('State.Name').count().reset_index()['State.Name'].tolist())

    with col2:
        # Filter for position
        subset_data = jobs
        select_position = st.multiselect(
        'Select position',
        jobs.groupby('Position').count().reset_index()['Position'].tolist())

    with col3:
        # Filter for employment type
        subset_data = jobs
        select_employment_type = st.multiselect(
        'Select employment type',
        jobs.groupby('Employment.Type').count().reset_index()['Employment.Type'].tolist())

        # by country name
        if len(select_state) > 0:
            subset_data = jobs[jobs['State.Name'].isin(select_state)]

    # inserting the map
    st.pydeck_chart(jobs_map)



# within tab3, recommender
with tab3:
    select_title = st.selectbox('Select job title', recommender_selectbox)
    n_recs = st.slider('How many recommendations?', 1, 5, 3)

    if st.button('Show me recommended jobs'):
        st.write(similar_title(select_title, n_recs), use_container_width=True)

# within tab4, recommender
with tab4:
    st.write('Collaborative recommender system based on application ID, job ID and duration maybe? (rows=applicant ID, columns=job ID and values=duration)')



# within tab5, the first row is divided into 5 columns included various metrics.
# this is followed by a multiselect box and a timeline showing the number of jobs posted per day/per employment type
# this is followed by bar charts showing the top 5 companies and positions posted
with tab5:
    st.markdown('#')
    
    # inserting various metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Open positions", open_positions, "+3%")
    col2.metric("Companies hiring", companies_hiring, "-1%")
    col3.metric("Active JobHunters", active_jobhunters, "+8%")
    col4.metric("Avg. job posts viewed per JobHunter", avg_job_posts_viewed_per_jobhunter, "-4%")
    col5.metric("Avg. viewtime per job post (min.)", avg_viewtime_per_job_post, "+7%")

    st.markdown('#')

        # showing the top 5 companies and positions posted
    col1, col2 = st.columns(2)

    with col1:
        st.write('#### Top 5 companies hiring right now')
        st.bar_chart(top_5_companies, use_container_width=True)
        

    with col2:
        st.write('#### Top 5 professions hiring right now')
        st.bar_chart(top_5_positions, use_container_width=True)

    # inserting multiselect boxes and timeline showing the number of jobs posted per day/per employment type

    y_axis_val = st.multiselect('The number of new job postings change throughout the year, check out the trends based on employment type below', options=jobs_pivot.columns)
    jobs_pivot_plot = px.line(jobs_pivot, y=y_axis_val)
    st.plotly_chart(jobs_pivot_plot, use_container_width=True)

    y_axis_val = st.multiselect('The number of new job postings change throughout the year, check out the trends based on position below', options=positions_pivot.columns)
    positions_pivot_plot = px.line(positions_pivot, y=y_axis_val)
    st.plotly_chart(positions_pivot_plot, use_container_width=True)



# within tab6, each 'row' is divided into 3 columns.
with tab6:
    # a header is created for the row. Each column in the row includes a title and a link to a youtube video.
    st.header('Tutorials ')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('##### How to write a powerful CV')
        st.video("https://www.youtube.com/watch?v=uG2aEh5xBJE")

    with col2:
        st.write('##### How to write a powerful cover letter')
        st.video("https://www.youtube.com/watch?v=lq6aGl1QBRs")
    
    with col3:
        st.write('##### How to prepare for a job interview')
        st.video("https://www.youtube.com/watch?v=enD8mK9Zvwo&list=RDCMUCIEU-iRzjXYo8JrOT9WGpnw&index=2")


    # a header is created for the row. Each column in the row includes a link to a CV template
    st.header('CV templates ')

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image('https://www.cvmaker.dk/assets/images/cvs/2/cv-example-harvard-3f6591.jpg')

    with col2:
        st.image('https://www.cvmaker.dk/assets/images/cvs/9/cv-example-edinburgh-505577.jpg')

    with col3:
        st.image('https://www.cvmaker.dk/assets/images/cvs/4/cv-example-cambridge-3f6592.jpg')


    # a header is created for the row. Each column in the row includes links to various tests and articles.
    st.header('Want to learn more?')

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.expander('Test Yourself'):
            st.write('- Test your personality type via this [link](https://www.jobindex.dk/persontypetest?lang=en)')
            st.write('- Test your job satisfaction via this [link](https://www.jobindex.dk/test-dig-selv/jobglaede?lang=en)')
            st.write('- Test your stress level via this [link](https://www.jobindex.dk/stress?lang=en)')
            st.write('- Test your talents via this [link](https://www.jobindex.dk/talenttest?lang=en)')
            st.write('- Test your salary via this [link](https://www.jobindex.dk/tjek-din-loen?lang=en)')

    with col2:
        with st.expander('Career development'):
            st.write('- Learn about career development via this [link](https://www.thebalancemoney.com/what-is-career-development-525496)')
            st.write('- Tips to improve career development via this [link](https://www.thebalancemoney.com/improving-career-development-4058289)')
            st.write('- Examine the benefits of mentoring via this [link](https://www.thebalancemoney.com/use-mentoring-to-develop-employees-1918189)')
            st.write('- Explore the concept of job-shadowing via this [link](https://www.thebalancemoney.com/job-shadowing-is-effective-on-the-job-training-1919285)')

    with col3:
        with st.expander('Guidance'):
            st.write('- Free career guidance via this [link](https://www.jobindex.dk/cms/jobvejledning?lang=en)')
            st.write('- Courses for unemployed via this [link](https://jobindexkurser.dk/kategori/alle-kategorier?lang=en)')

