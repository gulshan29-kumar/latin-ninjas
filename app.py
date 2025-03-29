import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from faker import Faker

# Initialize faker for generating realistic data
fake = Faker()

# Set page config
st.set_page_config(page_title="Employee Recommender", page_icon="👥", layout="wide")

# Skills database for generating realistic profiles
TECH_SKILLS = [
    'Python', 'Java', 'JavaScript', 'C#', 'C++', 'Ruby', 'Go', 'Swift', 'Kotlin',
    'SQL', 'NoSQL', 'Docker', 'Kubernetes', 'AWS', 'Azure', 'GCP', 'React', 'Angular',
    'Vue', 'Node.js', 'Django', 'Flask', 'Spring', 'TensorFlow', 'PyTorch', 'Pandas',
    'NumPy', 'Git', 'Jenkins', 'CI/CD', 'REST API', 'GraphQL', 'Microservices',
    'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP', 'Data Analysis',
    'Data Engineering', 'Data Science', 'Big Data', 'Hadoop', 'Spark', 'Tableau',
    'Power BI', 'UI/UX', 'Figma', 'Adobe XD', 'Cybersecurity', 'Ethical Hacking',
    'Blockchain', 'DevOps', 'SRE', 'Agile', 'Scrum', 'Project Management'
]

PROFESSIONS = [
    'Software Engineer', 'Data Scientist', 'Data Analyst', 'DevOps Engineer',
    'Frontend Developer', 'Backend Developer', 'Full Stack Developer',
    'Machine Learning Engineer', 'Cloud Architect', 'Security Engineer',
    'UX Designer', 'Product Manager', 'Technical Lead', 'Engineering Manager',
    'QA Engineer', 'Database Administrator', 'Systems Analyst', 'AI Specialist'
]

# Generate large employee dataset
def generate_employee_dataset(size=150):
    employees = []
    for _ in range(size):
        num_skills = random.randint(3, 8)
        skills = ', '.join(random.sample(TECH_SKILLS, num_skills))
        employees.append({
            'id': random.randint(1000, 9999),
            'name': fake.name(),
            'skills': skills,
            'profession': random.choice(PROFESSIONS),
            'email': fake.email(),
            'department': random.choice(['Engineering', 'Data', 'Product', 'Design', 'Security'])
        })
    return pd.DataFrame(employees)

# Initialize session state with large dataset
if 'employees' not in st.session_state:
    st.session_state.employees = generate_employee_dataset(150)  # Generate 150 employees

# Enhanced recommendation function
def recommend_employees(df, new_employee, top_n=10):
    # Combine skills with profession for better matching
    df['skills_profession'] = df['skills'] + ' ' + df['profession']
    new_skills_profession = new_employee['skills'] + ' ' + new_employee['profession']
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['skills_profession'].tolist() + [new_skills_profession])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:top_n]]
    
    recommendations = df.iloc[top_indices].copy()
    recommendations['Similarity Score'] = [f"{i[1]*100:.1f}%" for i in sim_scores[:top_n]]
    
    return recommendations[['name', 'profession', 'skills', 'email', 'department', 'Similarity Score']]

# Main app
def main():
    st.title("👥 Advanced Employee Recommendation System")
    st.markdown("Find your perfect teammates based on skills and expertise")
    
    # Display dataset stats
    st.sidebar.markdown(f"*Total Employees in Database:* {len(st.session_state.employees)}")
    
    # Enhanced filtering
    with st.expander("🔍 Explore Current Employees", expanded=False):
        department_filter = st.multiselect(
            "Filter by Department",
            options=st.session_state.employees['department'].unique()
        )
        
        filtered_df = st.session_state.employees
        if department_filter:
            filtered_df = filtered_df[filtered_df['department'].isin(department_filter)]
        
        st.dataframe(
            filtered_df,
            column_config={
                "name": "Name",
                "profession": "Profession",
                "skills": "Skills",
                "email": "Email",
                "department": "Department"
            },
            hide_index=True,
            use_container_width=True
        )
    
    # User input form
    with st.form("employee_form"):
        st.header("🧑‍💻 Add Your Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", placeholder="John Doe")
            profession = st.selectbox("Profession/Role*", options=PROFESSIONS)
            department = st.selectbox("Department", 
                                   options=['Engineering', 'Data', 'Product', 'Design', 'Security', 'Other'])
        with col2:
            skills = st.multiselect("Your Skills*", options=TECH_SKILLS)
            email = st.text_input("Email*", placeholder="your.email@company.com")
        
        submitted = st.form_submit_button("🚀 Find My Matches")
    
    # Process form submission
    if submitted:
        if not all([name, profession, skills, email]):
            st.error("Please fill in all required fields (marked with *)")
        else:
            new_emp = {
                'name': name,
                'profession': profession,
                'skills': ', '.join(skills),
                'email': email,
                'department': department
            }
            
            with st.spinner('Finding your best matches...'):
                recommendations = recommend_employees(st.session_state.employees, new_emp)
            
            st.success(f"Here are your top {len(recommendations)} recommended colleagues:")
            
            # Display recommendations
            st.dataframe(
                recommendations,
                column_config={
                    "name": "Name",
                    "profession": "Profession",
                    "skills": "Skills",
                    "email": "Email",
                    "department": "Department",
                    "Similarity Score": st.column_config.NumberColumn(
                        "Match Score",
                        help="Similarity based on skills and profession (0-100%)",
                        format="%.1f%%"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Enhanced visualization
            st.subheader("📊 Match Distribution")
            chart_data = recommendations.copy()
            chart_data['Score'] = chart_data['Similarity Score'].str.replace('%','').astype(float)
            st.bar_chart(chart_data.set_index('name')['Score'])
            
            # Add download button
            st.download_button(
                label="📥 Download Recommendations",
                data=recommendations.to_csv(index=False),
                file_name="employee_recommendations.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()