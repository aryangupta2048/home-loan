"""
🏠 Home Loan / EMI Advisor
---------------------------
A simple Streamlit app that helps a user:
1. Calculate their monthly EMI (Equated Monthly Installment)
2. Check whether they are likely eligible for the loan
3. View a month-by-month amortization (repayment) schedule
4. Get easy-to-understand loan advice from Google Gemini (AI)

This project is intentionally kept simple (plain functions, no classes,
no advanced patterns) so it is easy to explain in a college presentation.

Author: Student Project (RE-03)
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Home Loan / EMI Advisor", page_icon="🏠", layout="wide")


# ---------------------------------------------------------
# CORE FINANCE FUNCTIONS
# (These are plain Python functions - no external library needed
#  for the maths, so it is easy to explain to a class)
# ---------------------------------------------------------

def calculate_emi(principal, annual_rate, tenure_years):
    """
    Calculate the Equated Monthly Installment (EMI).

    Formula:
        EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)

    where:
        P = loan amount (principal)
        r = monthly interest rate (annual rate / 12 / 100)
        n = number of monthly installments (tenure in years * 12)
    """
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12

    if monthly_rate == 0:  # edge case: 0% interest loan
        return principal / months

    emi = principal * monthly_rate * (1 + monthly_rate) ** months
    emi = emi / ((1 + monthly_rate) ** months - 1)
    return emi


def generate_amortization_schedule(principal, annual_rate, tenure_years):
    """
    Build a month-wise repayment schedule showing how much of each
    EMI goes towards interest vs principal, and the remaining balance.
    Returns a pandas DataFrame.
    """
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    emi = calculate_emi(principal, annual_rate, tenure_years)

    balance = principal
    rows = []
    for month in range(1, months + 1):
        interest_payment = balance * monthly_rate
        principal_payment = emi - interest_payment
        balance = balance - principal_payment
        if balance < 0:
            balance = 0

        rows.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal Paid": round(principal_payment, 2),
            "Interest Paid": round(interest_payment, 2),
            "Remaining Balance": round(balance, 2)
        })

    return pd.DataFrame(rows)


def check_eligibility(monthly_income, existing_emis, requested_loan,
                       annual_rate, tenure_years, foir_limit=50):
    """
    A simplified loan-eligibility check based on FOIR
    (Fixed Obligation to Income Ratio) - a common real-world rule
    used by banks: total EMIs (old + new) should not exceed a
    certain percentage (commonly 40-50%) of monthly income.

    Returns a dictionary with the eligibility result and details.
    """
    max_allowed_emi = (foir_limit / 100) * monthly_income
    emi_available_for_new_loan = max_allowed_emi - existing_emis

    requested_emi = calculate_emi(requested_loan, annual_rate, tenure_years)

    is_eligible = requested_emi <= emi_available_for_new_loan and emi_available_for_new_loan > 0

    # Also estimate the maximum loan amount the person could get
    # with the remaining EMI budget (reverse EMI formula)
    monthly_rate = annual_rate / (12 * 100)
    months = tenure_years * 12
    if emi_available_for_new_loan > 0 and monthly_rate > 0:
        max_eligible_loan = emi_available_for_new_loan * ((1 + monthly_rate) ** months - 1)
        max_eligible_loan = max_eligible_loan / (monthly_rate * (1 + monthly_rate) ** months)
    elif emi_available_for_new_loan > 0:
        max_eligible_loan = emi_available_for_new_loan * months
    else:
        max_eligible_loan = 0

    return {
        "is_eligible": is_eligible,
        "requested_emi": round(requested_emi, 2),
        "max_allowed_emi": round(max_allowed_emi, 2),
        "emi_budget_left": round(emi_available_for_new_loan, 2),
        "max_eligible_loan": round(max_eligible_loan, 2)
    }


def ask_gemini_for_advice(api_key, profile_summary):
    """
    Send the user's loan profile to Google Gemini and get back
    simple, friendly, easy-to-understand loan advice.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    You are a friendly home loan advisor talking to a first-time home buyer.
    Based on the details below, give short, simple, practical advice
    (use plain language, avoid jargon, use bullet points, max 200 words).

    {profile_summary}

    Cover:
    - Whether the loan looks affordable for them
    - 2-3 tips to improve eligibility or reduce interest burden
    - One general home-loan tip for beginners
    """

    response = model.generate_content(prompt)
    return response.text


# ---------------------------------------------------------
# SIDEBAR - API KEY + USER INPUTS
# ---------------------------------------------------------
st.sidebar.title("⚙️ Settings")
api_key = st.sidebar.text_input("Enter your Google Gemini API Key", type="password")
st.sidebar.caption("Get a free key from https://aistudio.google.com/app/apikey")

st.sidebar.markdown("---")
st.sidebar.subheader("Your Details")

monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=0, value=60000, step=1000)
existing_emis = st.sidebar.number_input("Existing Monthly EMIs (₹)", min_value=0, value=0, step=500)

st.sidebar.markdown("---")
st.sidebar.subheader("Loan Details")
loan_amount = st.sidebar.number_input("Loan Amount Required (₹)", min_value=10000, value=2500000, step=10000)
interest_rate = st.sidebar.number_input("Annual Interest Rate (%)", min_value=1.0, value=8.5, step=0.1)
tenure_years = st.sidebar.slider("Loan Tenure (years)", min_value=1, max_value=30, value=20)


# ---------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------
st.title("🏠 Home Loan / EMI Advisor")
st.write("A simple tool to calculate your EMI, check loan eligibility, "
         "view your repayment schedule, and get AI-powered advice.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EMI Calculator",
    "✅ Eligibility Check",
    "📅 Amortization Schedule",
    "🤖 AI Recommendations"
])

# ---------------- TAB 1: EMI CALCULATOR ----------------
with tab1:
    st.subheader("EMI Calculator")

    emi = calculate_emi(loan_amount, interest_rate, tenure_years)
    total_payment = emi * tenure_years * 12
    total_interest = total_payment - loan_amount

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly EMI", f"₹ {emi:,.2f}")
    col2.metric("Total Interest Payable", f"₹ {total_interest:,.2f}")
    col3.metric("Total Payment (Principal + Interest)", f"₹ {total_payment:,.2f}")

    # Simple pie chart: principal vs interest
    fig, ax = plt.subplots()
    ax.pie(
        [loan_amount, total_interest],
        labels=["Principal", "Total Interest"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Principal vs Interest")
    st.pyplot(fig)

# ---------------- TAB 2: ELIGIBILITY CHECK ----------------
with tab2:
    st.subheader("Loan Eligibility Check")
    st.caption("Based on a simplified FOIR (Fixed Obligation to Income Ratio) rule "
               "used by many banks: your total EMIs should not exceed 50% of your income.")

    result = check_eligibility(monthly_income, existing_emis, loan_amount,
                                interest_rate, tenure_years)

    if result["is_eligible"]:
        st.success("✅ You are likely ELIGIBLE for this loan!")
    else:
        st.error("❌ You may NOT be eligible for this loan amount right now.")

    colA, colB = st.columns(2)
    with colA:
        st.write(f"**Required EMI for this loan:** ₹ {result['requested_emi']:,.2f}")
        st.write(f"**Max EMI you can afford (50% rule):** ₹ {result['max_allowed_emi']:,.2f}")
    with colB:
        st.write(f"**EMI budget left (after existing EMIs):** ₹ {result['emi_budget_left']:,.2f}")
        st.write(f"**Maximum loan you may be eligible for:** ₹ {result['max_eligible_loan']:,.2f}")

# ---------------- TAB 3: AMORTIZATION SCHEDULE ----------------
with tab3:
    st.subheader("Month-by-Month Repayment Schedule")

    schedule_df = generate_amortization_schedule(loan_amount, interest_rate, tenure_years)
    st.dataframe(schedule_df, use_container_width=True, height=350)

    # Line chart: how balance reduces over time
    fig2, ax2 = plt.subplots()
    ax2.plot(schedule_df["Month"], schedule_df["Remaining Balance"])
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Remaining Balance (₹)")
    ax2.set_title("Loan Balance Over Time")
    st.pyplot(fig2)

    csv = schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Schedule as CSV", csv, "amortization_schedule.csv", "text/csv")

# ---------------- TAB 4: AI RECOMMENDATIONS ----------------
with tab4:
    st.subheader("AI-Powered Loan Advice (Google Gemini)")

    if st.button("Get AI Recommendation"):
        if not api_key:
            st.warning("⚠️ Please enter your Gemini API key in the sidebar first.")
        else:
            profile_summary = f"""
            Monthly Income: ₹{monthly_income}
            Existing Monthly EMIs: ₹{existing_emis}
            Requested Loan Amount: ₹{loan_amount}
            Interest Rate: {interest_rate}%
            Tenure: {tenure_years} years
            Calculated EMI: ₹{emi:,.2f}
            Eligible: {result['is_eligible']}
            Max Eligible Loan: ₹{result['max_eligible_loan']:,.2f}
            """
            with st.spinner("Asking Gemini for advice..."):
                try:
                    advice = ask_gemini_for_advice(api_key, profile_summary)
                    st.markdown(advice)
                except Exception as e:
                    st.error(f"Something went wrong while calling Gemini API: {e}")
    else:
        st.info("Click the button above to get personalized, easy-to-understand loan advice.")

st.markdown("---")
st.caption("Project RE-03 | Home Loan / EMI Advisor | Built with Streamlit + Google Gemini API")
