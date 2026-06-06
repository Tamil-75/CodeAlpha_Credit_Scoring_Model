import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RiskHorizon | Credit Intelligence Portal", layout="wide", initial_sidebar_state="expanded", page_icon="\U0001f6e1\ufe0f")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&display=swap');
:root {
  --bg-p:#07090f; --bg-c:#0d1117; --bg-e:#161b27; --bg-h:#1c2336;
  --blu:#3b82f6; --blu-d:#2563eb; --teal:#06b6d4; --grn:#10b981;
  --red:#f43f5e; --amb:#f59e0b; --vio:#8b5cf6;
  --txt:#e2e8f0; --txt-m:#94a3b8; --txt-d:#475569;
  --brd:rgba(255,255,255,0.06); --brd-h:rgba(255,255,255,0.12);
}
html,body,[class*="css"],.main{font-family:'DM Sans',sans-serif!important;background-color:var(--bg-p)!important;color:var(--txt)!important;}
.block-container{padding:1.5rem 2.5rem 3rem!important;max-width:1400px!important;}
[data-testid="stSidebar"]{background:var(--bg-c)!important;border-right:1px solid var(--brd)!important;}
[data-testid="stSidebar"] *{color:var(--txt)!important;}
[data-testid="stMetric"]{background:var(--bg-c);border:1px solid var(--brd);border-radius:14px;padding:1.1rem 1.2rem!important;transition:border-color 0.2s;}
[data-testid="stMetric"]:hover{border-color:var(--brd-h);}
[data-testid="stMetricLabel"]{font-size:0.68rem!important;color:var(--txt-m)!important;text-transform:uppercase;letter-spacing:0.08em;}
[data-testid="stMetricValue"]{font-size:1.65rem!important;font-weight:700!important;font-family:'DM Mono',monospace!important;}
[data-testid="stMetricDelta"]{font-size:0.75rem!important;}
div.stButton>button:first-child{width:100%;background:linear-gradient(135deg,var(--blu) 0%,var(--teal) 100%);color:white;border-radius:10px;font-weight:600;border:none;padding:0.55rem 1rem;font-size:0.9rem;box-shadow:0 4px 16px rgba(59,130,246,0.25);transition:all 0.2s ease;}
div.stButton>button:first-child:hover{background:linear-gradient(135deg,var(--blu-d) 0%,#0891b2 100%);transform:translateY(-1px);box-shadow:0 6px 20px rgba(59,130,246,0.35);}
.stNumberInput input,.stTextInput input{background:var(--bg-e)!important;border:1px solid var(--brd)!important;border-radius:8px!important;color:var(--txt)!important;}
.stSelectbox>div>div{background:var(--bg-e)!important;border:1px solid var(--brd)!important;border-radius:8px!important;color:var(--txt)!important;}
[data-testid="stDataFrame"]{border:1px solid var(--brd)!important;border-radius:12px!important;overflow:hidden;background:var(--bg-c)!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--bg-c);border-radius:10px;padding:4px;border:1px solid var(--brd);}
.stTabs [data-baseweb="tab"]{border-radius:7px!important;padding:0.4rem 1rem!important;font-size:0.82rem!important;font-weight:500!important;color:var(--txt-m)!important;background:transparent!important;}
.stTabs [aria-selected="true"]{background:var(--bg-e)!important;color:var(--txt)!important;}
.badge{display:inline-block;padding:0.25rem 0.75rem;border-radius:999px;font-weight:600;font-size:0.75rem;letter-spacing:0.04em;text-transform:uppercase;}
.badge-grn{background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.25);}
.badge-amb{background:rgba(245,158,11,0.12);color:#fbbf24;border:1px solid rgba(245,158,11,0.25);}
.badge-red{background:rgba(244,63,94,0.12);color:#fb7185;border:1px solid rgba(244,63,94,0.25);}
.badge-vio{background:rgba(139,92,246,0.12);color:#a78bfa;border:1px solid rgba(139,92,246,0.25);}
.badge-blu{background:rgba(59,130,246,0.12);color:#60a5fa;border:1px solid rgba(59,130,246,0.25);}
.sec-label{font-size:0.62rem;font-weight:700;color:var(--teal);letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.2rem;}
.insight-card{background:var(--bg-c);border:1px solid var(--brd);border-radius:14px;padding:1.1rem 1.3rem;margin-bottom:0.6rem;border-left:3px solid var(--blu);}
.risk-flag{background:rgba(244,63,94,0.06);border:1px solid rgba(244,63,94,0.2);border-radius:10px;padding:0.7rem 1rem;margin-bottom:0.5rem;border-left:3px solid var(--red);font-size:0.83rem;color:var(--txt-m);}
hr{border-color:var(--brd)!important;margin:1rem 0!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-thumb{background:#2d3f5a;border-radius:4px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex;align-items:center;gap:16px;padding-bottom:0.5rem'>
  <div style='width:46px;height:46px;background:linear-gradient(135deg,#3b82f6,#06b6d4);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;flex-shrink:0'>🛡️</div>
  <div>
    <h1 style='margin:0;font-size:1.7rem;font-weight:700;letter-spacing:-0.02em;color:#e2e8f0'>
      RiskHorizon<sup style='font-size:0.55rem;vertical-align:super'>&trade;</sup> Intelligence Engine
    </h1>
    <p style='color:#475569;margin:2px 0 0;font-size:0.82rem'>Automated credit risk modeling &middot; Portfolio analytics &middot; Live underwriting</p>
  </div>
</div><hr>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='text-align:center;padding:0.5rem 0 1.2rem'>
  <div style='width:52px;height:52px;background:linear-gradient(135deg,#3b82f6,#06b6d4);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:1.6rem;margin:0 auto 8px'>🛡️</div>
  <div style='font-weight:700;font-size:1.05rem'>RiskHorizon&trade;</div>
  <div style='font-size:0.65rem;color:#475569;letter-spacing:0.08em;text-transform:uppercase;margin-top:2px'>Credit Intelligence v3.0</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**\U0001f4c2 Data Sync**")
uploaded_file = st.sidebar.file_uploader("Upload credit register CSV", type=["csv"])

if uploaded_file is None:
    st.markdown("""
    <div style='background:#0d1117;border:1px dashed rgba(59,130,246,0.25);border-radius:16px;padding:3rem 2rem;text-align:center;margin-top:2rem'>
      <div style='font-size:3rem;margin-bottom:0.8rem'>📊</div>
      <div style='font-weight:600;font-size:1.1rem;color:#e2e8f0;margin-bottom:0.4rem'>Awaiting Data Feed</div>
      <div style='color:#475569;font-size:0.85rem'>Upload <code style='background:#1e293b;padding:2px 8px;border-radius:5px;color:#38bdf8'>credit_risk_dataset.csv</code> to activate the analytics engine.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

GRADE_MAP  = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
HOME_MAP   = {'RENT':0,'OWN':1,'MORTGAGE':2,'OTHER':3}
INTENT_MAP = {'PERSONAL':0,'EDUCATION':1,'MEDICAL':2,'VENTURE':3,'HOMEIMPROVEMENT':4,'DEBTCONSOLIDATION':5}

@st.cache_data
def process_data(file):
    df = pd.read_csv(file)
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df.loc[df['person_emp_length'] > 60, 'person_emp_length'] = df['person_emp_length'].median()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    df['Monthly_Income']         = (df['person_income'] / 12).round(2)
    df['Debt_to_Income_Ratio']   = df['loan_percent_income']
    df['Prior_Default']          = (df['cb_person_default_on_file'] == 'Y').astype(int)
    df['Loan_Grade_Encoded']     = df['loan_grade'].map(GRADE_MAP)
    df['Home_Ownership_Encoded'] = df['person_home_ownership'].map(HOME_MAP).fillna(3)
    df['Loan_Intent_Encoded']    = df['loan_intent'].str.upper().map(INTENT_MAP).fillna(0)
    df['Income_to_Loan_Ratio']   = (df['person_income'] / df['loan_amnt'].replace(0, np.nan)).round(3)
    return df.rename(columns={'loan_status': 'Default'})

df = process_data(uploaded_file)
st.sidebar.success("\u2705 Analytics Engine Active")
st.sidebar.markdown(f"<div style='font-size:0.75rem;color:#475569;margin-top:-8px'>{len(df):,} records loaded</div>", unsafe_allow_html=True)

FEATURES = [
    'person_age','person_income','person_emp_length',
    'loan_amnt','loan_int_rate','Monthly_Income','Debt_to_Income_Ratio',
    'cb_person_cred_hist_length','Prior_Default',
    'Loan_Grade_Encoded','Home_Ownership_Encoded',
    'Loan_Intent_Encoded','Income_to_Loan_Ratio'
]

@st.cache_data
def train_all_models(n_rows, _df):
    X, y = _df[FEATURES], _df['Default']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200,max_depth=12,min_samples_leaf=4,class_weight='balanced_subsample',random_state=42,n_jobs=-1),
        'Gradient Boost': GradientBoostingClassifier(n_estimators=150,max_depth=5,learning_rate=0.08,subsample=0.8,random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42,C=0.5)
    }
    results = {}
    for name,m in models.items():
        m.fit(X_tr_s,y_train)
        probs = m.predict_proba(X_te_s)[:,1]
        preds = m.predict(X_te_s)
        fpr,tpr,_ = roc_curve(y_test,probs)
        results[name] = {'model':m,'report':classification_report(y_test,preds,output_dict=True),'auc':roc_auc_score(y_test,probs),'cm':confusion_matrix(y_test,preds),'fpr':fpr,'tpr':tpr}
    return results,scaler

results,scaler = train_all_models(len(df),df)
best_model_name = max(results,key=lambda k:results[k]['auc'])

st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("**\U0001f9ed Navigation**")
option = st.sidebar.radio("",["\U0001f5c2\ufe0f Portfolio Explorer","\U0001f3af Credit Scorecard","\U0001f4ca Model Analytics","\U0001f52c Segment Intelligence","\u2696\ufe0f Model Comparison"],label_visibility="collapsed")
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.markdown("**\U0001f916 Active Model**")
selected_model_name = st.sidebar.selectbox("",list(results.keys()),index=list(results.keys()).index(best_model_name),label_visibility="collapsed")
active       = results[selected_model_name]
active_model = active['model']
auc          = active['auc']
report       = active['report']
cm           = active['cm']

DT = dict(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="#94a3b8",size=12),margin=dict(t=40,b=30,l=30,r=30))

# ── PAGE 1 ──────────────────────────────────────────────────────────────────
if option == "\U0001f5c2\ufe0f Portfolio Explorer":
    st.markdown("<div class='sec-label'>Portfolio Audit</div>", unsafe_allow_html=True)
    st.subheader("\U0001f5c2\ufe0f Portfolio Ledger & Distribution")
    def_rate = round((df['Default'].sum()/len(df))*100,2)
    avg_loan = df['loan_amnt'].mean()
    prior_def_pct = round((df['Prior_Default'].sum()/len(df))*100,2)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Accounts",f"{len(df):,}")
    c2.metric("Portfolio Default Rate",f"{def_rate}%",delta="High Risk" if def_rate>20 else "Controlled",delta_color="inverse")
    c3.metric("Avg Loan Amount",f"${avg_loan:,.0f}")
    c4.metric("Prior Default on File",f"{prior_def_pct}%")
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander("\U0001f50d Filter & Explore Records",expanded=True):
        fc1,fc2,fc3 = st.columns(3)
        grade_filter  = fc1.multiselect("Loan Grade",sorted(df['loan_grade'].unique()),default=sorted(df['loan_grade'].unique()))
        intent_filter = fc2.multiselect("Loan Intent",sorted(df['loan_intent'].unique()),default=sorted(df['loan_intent'].unique()))
        def_filter    = fc3.selectbox("Default Status",["All","Defaulted","Performing"])
        filtered = df[df['loan_grade'].isin(grade_filter)&df['loan_intent'].isin(intent_filter)]
        if def_filter=="Defaulted":    filtered=filtered[filtered['Default']==1]
        elif def_filter=="Performing": filtered=filtered[filtered['Default']==0]
        st.caption(f"Showing {len(filtered):,} of {len(df):,} records")
        st.dataframe(filtered[['person_age','person_income','loan_grade','loan_intent','loan_amnt','loan_int_rate','Debt_to_Income_Ratio','cb_person_cred_hist_length','Prior_Default','Default']].head(20).style.format({'person_income':'${:,.0f}','loan_amnt':'${:,.0f}','loan_int_rate':'{:.1f}%','Debt_to_Income_Ratio':'{:.1%}'}),use_container_width=True,height=300)
    st.markdown("<hr>", unsafe_allow_html=True)
    ch1,ch2 = st.columns(2)
    counts = df['Default'].value_counts().reset_index()
    counts.columns=['Status','Count']
    counts['Status']=counts['Status'].apply(lambda x:'Default' if str(x) in ['1','1.0'] else 'Performing')
    fig1=px.pie(counts,values='Count',names='Status',title='Default vs Performing',color='Status',color_discrete_map={'Performing':'#10b981','Default':'#f43f5e'},hole=0.6)
    fig1.update_layout(**DT); fig1.update_traces(marker=dict(line=dict(color='#07090f',width=2)))
    ch1.plotly_chart(fig1,use_container_width=True)
    grade_def=df.groupby('loan_grade')['Default'].mean().reset_index()
    grade_def.columns=['Grade','Default Rate']
    grade_def=grade_def.sort_values('Grade')
    fig2=px.bar(grade_def,x='Grade',y='Default Rate',title='Default Rate by Loan Grade',color='Default Rate',color_continuous_scale=['#10b981','#f59e0b','#f43f5e'])
    fig2.update_layout(**DT); fig2.update_coloraxes(showscale=False); fig2.update_yaxes(tickformat='.0%')
    ch2.plotly_chart(fig2,use_container_width=True)
    ch3,ch4=st.columns(2)
    fig3=px.histogram(df,x='loan_int_rate',color='Default',barmode='overlay',nbins=40,title='Interest Rate Distribution by Default Status',color_discrete_map={0:'#3b82f6',1:'#f43f5e'})
    fig3.update_layout(**DT); fig3.update_traces(opacity=0.75)
    ch3.plotly_chart(fig3,use_container_width=True)
    intent_def=df.groupby('loan_intent')['Default'].agg(['mean','count']).reset_index()
    intent_def.columns=['Intent','Default Rate','Count']
    fig4=px.bar(intent_def.sort_values('Default Rate',ascending=True),x='Default Rate',y='Intent',orientation='h',title='Default Rate by Loan Intent',color='Default Rate',color_continuous_scale=['#10b981','#f59e0b','#f43f5e'])
    fig4.update_layout(**DT); fig4.update_coloraxes(showscale=False); fig4.update_xaxes(tickformat='.0%')
    ch4.plotly_chart(fig4,use_container_width=True)

# ── PAGE 2 ──────────────────────────────────────────────────────────────────
elif option == "\U0001f3af Credit Scorecard":
    st.markdown("<div class='sec-label'>Live Underwriting</div>", unsafe_allow_html=True)
    st.subheader("\U0001f464 Real-Time Credit Scorecard")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown("<div style='font-size:0.73rem;color:#64748b;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em'> Applicant Profile</div>", unsafe_allow_html=True)
        age      = st.slider("Age (Years)",18,80,28)
        income   = st.number_input("Annual Income ($)",min_value=5000,max_value=1_000_000,value=65000,step=2500)
        emp_len  = st.slider("Employment Length (Years)",0,40,5)
        home_own = st.selectbox("Home Ownership",["RENT","MORTGAGE","OWN","OTHER"])
    with col2:
        st.markdown("<div style='font-size:0.73rem;color:#64748b;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em'>Loan Details</div>", unsafe_allow_html=True)
        loan_amt    = st.number_input("Loan Amount ($)",min_value=500,max_value=500_000,value=14000,step=500)
        int_rate    = st.slider("Interest Rate (%)",4.0,28.0,11.5,step=0.1)
        loan_grade  = st.selectbox("Loan Grade",["A","B","C","D","E","F","G"])
        loan_intent = st.selectbox("Loan Intent",["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"])
    with col3:
        st.markdown("<div style='font-size:0.73rem;color:#64748b;font-weight:600;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.06em'>Credit Bureau</div>", unsafe_allow_html=True)
        cred_hist     = st.slider("Credit History Length (Years)",0,30,4)
        prior_default = st.selectbox("Prior Default on File",["No","Yes"])
        dti         = round(loan_amt/income,4) if income>0 else 0.0
        inc_to_loan = round(income/loan_amt,3) if loan_amt>0 else 0.0
        st.markdown(f"<div style='background:var(--bg-e);border:1px solid var(--brd);border-radius:10px;padding:0.8rem 1rem;margin-top:0.5rem'><div style='font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px'>Computed Ratios</div><div style='display:flex;justify-content:space-between'><div><div style='font-size:0.68rem;color:#94a3b8'>Debt-to-Income</div><div style='font-weight:600;font-family:&quot;DM Mono&quot;,monospace;color:#e2e8f0'>{dti:.1%}</div></div><div><div style='font-size:0.68rem;color:#94a3b8'>Income / Loan</div><div style='font-weight:600;font-family:&quot;DM Mono&quot;,monospace;color:#e2e8f0'>{inc_to_loan:.2f}x</div></div></div></div>", unsafe_allow_html=True)
    if loan_amt > income*1.5: st.warning("\u26a0\ufe0f Requested principal significantly exceeds annual income. High leverage alert.")
    if prior_default=="Yes":  st.error("\U0001f6a8 Prior default on bureau file detected. Elevated scrutiny required.")
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("\U0001f50d Run Algorithmic Risk Assessment"):
        inp = pd.DataFrame([{'person_age':age,'person_income':income,'person_emp_length':emp_len,'loan_amnt':loan_amt,'loan_int_rate':int_rate,'Monthly_Income':round(income/12,2),'Debt_to_Income_Ratio':dti,'cb_person_cred_hist_length':cred_hist,'Prior_Default':1 if prior_default=="Yes" else 0,'Loan_Grade_Encoded':GRADE_MAP[loan_grade],'Home_Ownership_Encoded':HOME_MAP.get(home_own,3),'Loan_Intent_Encoded':INTENT_MAP.get(loan_intent,0),'Income_to_Loan_Ratio':inc_to_loan}])
        prob  = active_model.predict_proba(scaler.transform(inp))[0][1]
        cibil = int(300+(1-prob)*600)
        if cibil>=750:   tier,b_cls,col,rec="EXCELLENT \u2014 PRIME PLUS","badge-grn","#10b981","Profile is exceptionally secure. Auto-approved with best pricing."
        elif cibil>=650: tier,b_cls,col,rec="GOOD \u2014 PRIME","badge-blu","#3b82f6","Moderate risk. Standard approval. Secondary review recommended."
        elif cibil>=550: tier,b_cls,col,rec="AVERAGE \u2014 SUB-PRIME","badge-amb","#f59e0b","Elevated risk. Consider co-signer or collateral requirement."
        else:            tier,b_cls,col,rec="POOR \u2014 CRITICAL HAZARD","badge-red","#f43f5e","High default liability. Facility declined per core risk policy."
        st.markdown("<hr>", unsafe_allow_html=True)
        r1,r2,r3,r4=st.columns(4)
        r1.metric("Credit Score",f"{cibil} / 900")
        r2.metric("Default Probability",f"{prob:.1%}")
        r3.metric("Decision","\u2705 Approved" if cibil>=650 else "\u274c Declined")
        r4.metric("Active Model",selected_model_name.split()[0])
        st.markdown(f"<div style='margin:1rem 0'><span class='badge {b_cls}'>{tier}</span><span style='color:#64748b;font-size:0.83rem;margin-left:10px'>{rec}</span></div>", unsafe_allow_html=True)
        fig_g=go.Figure(go.Indicator(mode="gauge+number",value=cibil,title={'text':f"Bureau Scorecard \u2014 {selected_model_name}",'font':{'size':13,'color':'#94a3b8'}},number={'font':{'size':34,'color':col}},gauge={'axis':{'range':[300,900],'tickcolor':'#475569','nticks':7},'bar':{'color':col,'thickness':0.25},'bgcolor':'#1c2336','borderwidth':1,'bordercolor':'rgba(255,255,255,0.08)','steps':[{'range':[300,550],'color':'rgba(244,63,94,0.10)'},{'range':[550,650],'color':'rgba(245,158,11,0.10)'},{'range':[650,750],'color':'rgba(59,130,246,0.10)'},{'range':[750,900],'color':'rgba(16,185,129,0.10)'}],'threshold':{'line':{'color':col,'width':3},'thickness':0.8,'value':cibil}}))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font=dict(family="DM Sans",color="#94a3b8"),height=240,margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_g,use_container_width=True)
        st.markdown("<p style='font-size:0.83rem;font-weight:600;margin:0.5rem 0 0.4rem'>\u26a0\ufe0f Risk Anchors:</p>", unsafe_allow_html=True)
        flags=[]
        if dti>0.35:             flags.append(("Excessive Leverage",f"Debt-to-Income of {dti:.1%} exceeds 35% safe threshold."))
        if int_rate>15.0:        flags.append(("High Interest Rate",f"Rate of {int_rate:.1f}% signals elevated credit risk tier."))
        if prior_default=="Yes": flags.append(("Bureau Default History","Prior default on file is the strongest predictor of future default."))
        if loan_grade in list('DEFG'): flags.append(("Sub-Prime Loan Grade",f"Grade {loan_grade} places this facility in the high-risk tier."))
        if cred_hist<3:          flags.append(("Thin Credit File",f"Only {cred_hist} year(s) of history \u2014 insufficient track record."))
        if inc_to_loan<1.5:      flags.append(("Low Income Coverage",f"Income-to-loan ratio of {inc_to_loan:.2f}x is below safe minimum of 1.5x."))
        if flags:
            for title,desc in flags: st.markdown(f"<div class='risk-flag'><b style='color:#fb7185'>{title}:</b> {desc}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='insight-card' style='border-left-color:#10b981'>\u2705 No significant risk anchors detected.</div>", unsafe_allow_html=True)
        report_txt=f"""==================================================\nRISKHORIZON\u2122 COMPLIANCE AUDIT LOG v3.0\n==================================================\nModel Used             : {selected_model_name}\n--------------------------------------------------\nApplicant Age          : {age}\nAnnual Income          : ${income:,.2f}\nEmployment Length      : {emp_len} years\nHome Ownership         : {home_own}\n--------------------------------------------------\nLoan Amount            : ${loan_amt:,.2f}\nInterest Rate          : {int_rate:.2f}%\nLoan Grade             : {loan_grade}\nLoan Intent            : {loan_intent}\n--------------------------------------------------\nCredit History         : {cred_hist} years\nPrior Default on File  : {prior_default}\nDebt-to-Income         : {dti:.2%}\nIncome / Loan Ratio    : {inc_to_loan:.3f}x\n--------------------------------------------------\nCIBIL Score            : {cibil} / 900\nProbability of Default : {prob:.4f}\nEngine Decision        : {'APPROVED' if cibil>=650 else 'DECLINED'}\nRisk Classification    : {tier}\n==================================================\n{rec}\n=================================================="""
        st.download_button(label="\U0001f4e5 Export Underwriting Audit Log",data=report_txt,file_name=f"RiskHorizon_Audit_{cibil}.txt",mime="text/plain")

# ── PAGE 3 ──────────────────────────────────────────────────────────────────
elif option == "\U0001f4ca Model Analytics":
    st.markdown("<div class='sec-label'>Model Verification</div>", unsafe_allow_html=True)
    st.subheader(f"\U0001f4ca Performance Dashboard \u2014 {selected_model_name}")
    m1,m2,m3,m4=st.columns(4)
    m1.metric("ROC-AUC Score",f"{auc:.4f}",delta="Best" if selected_model_name==best_model_name else None)
    m2.metric("Precision (Default)",f"{report['1']['precision']:.4f}")
    m3.metric("Recall (Sensitivity)",f"{report['1']['recall']:.4f}")
    m4.metric("F1-Score",f"{report['1']['f1-score']:.4f}")
    st.markdown("<hr>", unsafe_allow_html=True)
    tab1,tab2,tab3=st.tabs(["\U0001f4c8 ROC Curve","\U0001f7e6 Confusion Matrix","\U0001f3af Feature Importance"])
    with tab1:
        fig_roc=go.Figure()
        roc_colors={'Random Forest':'#3b82f6','Gradient Boost':'#06b6d4','Logistic Regression':'#8b5cf6'}
        for name,res in results.items():
            fig_roc.add_trace(go.Scatter(x=res['fpr'],y=res['tpr'],mode='lines',name=f"{name} (AUC={res['auc']:.3f})",line=dict(color=roc_colors.get(name,'#94a3b8'),width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',line=dict(color='#475569',dash='dash',width=1.5),name='Random Baseline'))
        fig_roc.update_layout(title="ROC Curves \u2014 All Models",xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",**DT,height=420)
        st.plotly_chart(fig_roc,use_container_width=True)
    with tab2:
        labels=['Performing','Default']
        fig_cm=go.Figure(data=go.Heatmap(z=cm,x=labels,y=labels,colorscale=[[0,'#0d1117'],[1,'#3b82f6']],text=cm,texttemplate="<b>%{text:,}</b>",showscale=False))
        tn,fp,fn,tp=cm.ravel()
        fig_cm.update_layout(title=f"Confusion Matrix \u2014 {selected_model_name}",xaxis_title="Predicted",yaxis_title="Actual",**DT,height=380)
        st.plotly_chart(fig_cm,use_container_width=True)
        cc1,cc2,cc3,cc4=st.columns(4)
        cc1.metric("True Negatives",f"{tn:,}"); cc2.metric("False Positives",f"{fp:,}")
        cc3.metric("False Negatives",f"{fn:,}"); cc4.metric("True Positives",f"{tp:,}")
    with tab3:
        if hasattr(active_model,'feature_importances_'):
            fi=pd.DataFrame({'Feature':FEATURES,'Importance':active_model.feature_importances_}).sort_values('Importance',ascending=True)
            fig_imp=px.bar(fi,x='Importance',y='Feature',orientation='h',title=f'Feature Importance \u2014 {selected_model_name}',color='Importance',color_continuous_scale=['#1e3a5f','#3b82f6','#06b6d4'])
            fig_imp.update_layout(**DT,height=500,showlegend=False); fig_imp.update_coloraxes(showscale=False)
            st.plotly_chart(fig_imp,use_container_width=True)
        elif hasattr(active_model,'coef_'):
            coef_df=pd.DataFrame({'Feature':FEATURES,'Coefficient':active_model.coef_[0]})
            coef_df['Abs']=coef_df['Coefficient'].abs()
            coef_df=coef_df.sort_values('Abs',ascending=True)
            fig_coef=px.bar(coef_df,x='Coefficient',y='Feature',orientation='h',title='Logistic Regression Coefficients',color='Coefficient',color_continuous_scale=['#10b981','#1e293b','#f43f5e'])
            fig_coef.update_layout(**DT,height=500)
            st.plotly_chart(fig_coef,use_container_width=True)

# ── PAGE 4 ──────────────────────────────────────────────────────────────────
elif option == "\U0001f52c Segment Intelligence":
    st.markdown("<div class='sec-label'>Segment Drill-Down</div>", unsafe_allow_html=True)
    st.subheader("\U0001f52c Portfolio Segment Intelligence")
    sc1,sc2=st.columns(2)
    seg_by=sc1.selectbox("Segment by",['loan_grade','loan_intent','person_home_ownership','cb_person_default_on_file'])
    seg_metric=sc2.radio("Metric",['Default Rate','Avg Loan Amount','Avg Interest Rate','Count'],horizontal=True)
    metric_map={'Default Rate':('Default','mean'),'Avg Loan Amount':('loan_amnt','mean'),'Avg Interest Rate':('loan_int_rate','mean'),'Count':('Default','count')}
    col_agg,agg_fn=metric_map[seg_metric]
    seg_df=df.groupby(seg_by)[col_agg].agg(agg_fn).reset_index()
    seg_df.columns=['Segment',seg_metric]
    seg_df=seg_df.sort_values(seg_metric,ascending=False)
    fig_seg=px.bar(seg_df,x='Segment',y=seg_metric,title=f'{seg_metric} by {seg_by}',color=seg_metric,color_continuous_scale=['#1d9e75','#f59e0b','#f43f5e'])
    if seg_metric=='Default Rate': fig_seg.update_yaxes(tickformat='.0%')
    elif seg_metric=='Avg Loan Amount': fig_seg.update_yaxes(tickprefix='$',tickformat=',.0f')
    fig_seg.update_layout(**DT,height=380); fig_seg.update_coloraxes(showscale=False)
    st.plotly_chart(fig_seg,use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    df['Age_Band']=pd.cut(df['person_age'],bins=[17,25,35,45,55,65,100],labels=['18-25','26-35','36-45','46-55','56-65','65+'])
    age_def=df.groupby('Age_Band',observed=True)['Default'].agg(['mean','count']).reset_index()
    age_def.columns=['Age Band','Default Rate','Count']
    ch1,ch2=st.columns(2)
    fig_age=px.bar(age_def,x='Age Band',y='Default Rate',title='Default Rate by Age Band',color='Default Rate',color_continuous_scale=['#1d9e75','#f59e0b','#f43f5e'],text='Count')
    fig_age.update_yaxes(tickformat='.0%'); fig_age.update_traces(texttemplate='n=%{text:,}',textposition='outside')
    fig_age.update_layout(**DT,height=340); fig_age.update_coloraxes(showscale=False)
    ch1.plotly_chart(fig_age,use_container_width=True)
    df['Income_Band']=pd.cut(df['person_income'],bins=[0,30000,60000,100000,200000,1e9],labels=['<30k','30-60k','60-100k','100-200k','200k+'])
    inc_def=df.groupby('Income_Band',observed=True)['Default'].mean().reset_index()
    inc_def.columns=['Income Band','Default Rate']
    fig_inc=px.line(inc_def,x='Income Band',y='Default Rate',title='Default Rate by Income Bracket',markers=True,line_shape='spline',color_discrete_sequence=['#3b82f6'])
    fig_inc.update_yaxes(tickformat='.0%'); fig_inc.update_layout(**DT,height=340)
    ch2.plotly_chart(fig_inc,use_container_width=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**\U0001f4d0 Feature Correlation Matrix**")
    num_cols=['person_age','person_income','loan_amnt','loan_int_rate','Debt_to_Income_Ratio','cb_person_cred_hist_length','Prior_Default','Loan_Grade_Encoded','Default']
    corr=df[num_cols].corr().round(2)
    fig_corr=go.Figure(data=go.Heatmap(z=corr.values,x=num_cols,y=num_cols,colorscale='RdBu',zmid=0,text=corr.values,texttemplate="%{text:.2f}",showscale=True,colorbar=dict(thickness=12)))
    fig_corr.update_layout(title="Pearson Correlation \u2014 Key Features",**DT,height=480)
    st.plotly_chart(fig_corr,use_container_width=True)

# ── PAGE 5 ──────────────────────────────────────────────────────────────────
elif option == "\u2696\ufe0f Model Comparison":
    st.markdown("<div class='sec-label'>Benchmarking</div>", unsafe_allow_html=True)
    st.subheader("\u2696\ufe0f Model Comparison & Benchmarking")
    rows=[]
    for name,res in results.items():
        rows.append({'Model':name,'AUC-ROC':round(res['auc'],4),'Precision':round(res['report']['1']['precision'],4),'Recall':round(res['report']['1']['recall'],4),'F1-Score':round(res['report']['1']['f1-score'],4),'Accuracy':round(res['report']['accuracy'],4)})
    cmp_df=pd.DataFrame(rows).set_index('Model')
    def highlight_best(s):
        is_best=s==s.max()
        return ['background-color:rgba(16,185,129,0.15);color:#34d399' if v else '' for v in is_best]
    st.dataframe(cmp_df.style.apply(highlight_best),use_container_width=True,height=160)
    best_auc=results[best_model_name]['auc']
    st.markdown(f"<div class='insight-card'><b style='color:#60a5fa'>\U0001f3c6 Best Model: {best_model_name}</b> \u2014 AUC of {best_auc:.4f}. Green cells show top performer per metric.</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    ch1,ch2=st.columns(2)
    metrics=['AUC-ROC','Precision','Recall','F1-Score','Accuracy']
    fig_radar=go.Figure()
    radar_colors=['#3b82f6','#06b6d4','#8b5cf6']
    for i,(name,res) in enumerate(results.items()):
        vals=[round(res['auc'],4),round(res['report']['1']['precision'],4),round(res['report']['1']['recall'],4),round(res['report']['1']['f1-score'],4),round(res['report']['accuracy'],4)]
        fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=metrics+[metrics[0]],fill='toself',name=name,line=dict(color=radar_colors[i],width=2),opacity=0.75))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.5,1.0],color='#475569'),angularaxis=dict(color='#475569')),title="Performance Radar",**DT,height=420)
    ch1.plotly_chart(fig_radar,use_container_width=True)
    cmp_melt=cmp_df.reset_index().melt(id_vars='Model',var_name='Metric',value_name='Score')
    fig_bar=px.bar(cmp_melt,x='Metric',y='Score',color='Model',barmode='group',title='Metric Comparison',color_discrete_map={'Random Forest':'#3b82f6','Gradient Boost':'#06b6d4','Logistic Regression':'#8b5cf6'})
    fig_bar.update_layout(**DT,height=420,yaxis=dict(range=[0.5,1.0]))
    ch2.plotly_chart(fig_bar,use_container_width=True)