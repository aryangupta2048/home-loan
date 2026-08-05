"""
Home Loan / EMI Advisor
========================
RE-03 | Intermediate | Chain + Financial Calculator Tool

A Streamlit app that combines a custom-built financial calculator
(EMI, amortization, eligibility) with a LangChain chain that turns
the raw numbers into a personalized, plain-English loan recommendation.

Run:
    streamlit run app.py
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# LangChain is optional at import time — the app still works (in rule-based
# mode) if it isn't installed or no API key is supplied.
# ----------------------------------------------------------------------------
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False


# =============================================================================
# 1. CUSTOM FINANCIAL CALCULATOR TOOL
#    (Plain Python — this is the "financial calculator tool" the chain wraps)
# =============================================================================
class LoanCalculator:
    """Custom calculator tool for EMI, amortization and eligibility maths."""

    @staticmethod
    def monthly_rate(annual_rate_pct: float) -> float:
        return (annual_rate_pct / 100) / 12

    @classmethod
    def calculate_emi(cls, principal: float, annual_rate_pct: float, tenure_years: float) -> float:
        """Standard reducing-balance EMI formula."""
        r = cls.monthly_rate(annual_rate_pct)
        n = int(round(tenure_years * 12))
        if r == 0:
            return principal / n
        emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
        return emi

    @classmethod
    def amortization_schedule(cls, principal: float, annual_rate_pct: float, tenure_years: float) -> pd.DataFrame:
        r = cls.monthly_rate(annual_rate_pct)
        n = int(round(tenure_years * 12))
        emi = cls.calculate_emi(principal, annual_rate_pct, tenure_years)

        balance = principal
        rows = []
        for month in range(1, n + 1):
            interest_component = balance * r
            principal_component = emi - interest_component
            balance = max(balance - principal_component, 0)
            rows.append(
                {
                    "Month": month,
                    "Year": math.ceil(month / 12),
                    "EMI": round(emi, 2),
                    "Principal Paid": round(principal_component, 2),
                    "Interest Paid": round(interest_component, 2),
                    "Remaining Balance": round(balance, 2),
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def max_eligible_loan(
        cls,
        monthly_income: float,
        existing_emis: float,
        annual_rate_pct: float,
        tenure_years: float,
        foir_limit_pct: float = 50.0,
    ) -> dict:
        """
        Reverse-engineers the maximum loan a bank would typically sanction,
        based on the Fixed Obligation to Income Ratio (FOIR) banks commonly use.
        """
        max_allowed_emi = max(monthly_income * (foir_limit_pct / 100) - existing_emis, 0)
        r = cls.monthly_rate(annual_rate_pct)
        n = int(round(tenure_years * 12))

        if r == 0:
            max_principal = max_allowed_emi * n
        else:
            max_principal = max_allowed_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)

        return {
            "max_allowed_emi": round(max_allowed_emi, 2),
            "max_eligible_loan": round(max_principal, 2),
            "foir_limit_pct": foir_limit_pct,
        }

    @classmethod
    def eligibility_check(
        cls,
        requested_loan: float,
        monthly_income: float,
        existing_emis: float,
        annual_rate_pct: float,
        tenure_years: float,
        credit_score: int,
        foir_limit_pct: float = 50.0,
    ) -> dict:
        requested_emi = cls.calculate_emi(requested_loan, annual_rate_pct, tenure_years)
        elig = cls.max_eligible_loan(monthly_income, existing_emis, annual_rate_pct, tenure_years, foir_limit_pct)
        current_foir = ((existing_emis + requested_emi) / monthly_income * 100) if monthly_income else 0

        is_eligible = (requested_loan <= elig["max_eligible_loan"]) and (credit_score >= 650)

        # Reasons, useful for both the UI and the chain prompt
        reasons = []
        if requested_loan > elig["max_eligible_loan"]:
            reasons.append("Requested loan exceeds the FOIR-based maximum eligible amount.")
        if credit_score < 650:
            reasons.append("Credit score is below the commonly required threshold of 650.")
        if not reasons:
            reasons.append("Requested loan and EMI comfortably fit within income and credit norms.")

        return {
            "requested_emi": round(requested_emi, 2),
            "current_foir_pct": round(current_foir, 2),
            "is_eligible": is_eligible,
            "max_eligible_loan": elig["max_eligible_loan"],
            "max_allowed_emi": elig["max_allowed_emi"],
            "reasons": reasons,
        }


# =============================================================================
# 2. CHAIN — wraps the calculator's numeric output into a natural-language
#    recommendation. Uses LangChain LCEL: calculator -> prompt -> LLM -> text
# =============================================================================
RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful, neutral home-loan advisor. You are given structured "
            "financial figures already computed by a calculator tool. Do not "
            "recompute numbers yourself — just interpret them. Give a short, "
            "friendly, 4-6 sentence recommendation. Mention eligibility status, "
            "the FOIR, and 1-2 concrete next steps (e.g. reduce loan amount, "
            "extend tenure, improve credit score, add a co-applicant). Avoid "
            "guaranteeing loan approval since that is the bank's decision.",
        ),
        (
            "human",
            "Applicant snapshot:\n"
            "- Requested loan amount: ₹{loan_amount:,.0f}\n"
            "- Tenure: {tenure_years} years at {rate}% annual interest\n"
            "- Monthly income: ₹{income:,.0f}\n"
            "- Existing EMI obligations: ₹{existing_emis:,.0f}\n"
            "- Credit score: {credit_score}\n\n"
            "Calculator results:\n"
            "- Requested EMI: ₹{requested_emi:,.0f}\n"
            "- Resulting FOIR: {foir}%\n"
            "- Eligible: {eligible}\n"
            "- Max eligible loan at this rate/tenure: ₹{max_loan:,.0f}\n"
            "- Max allowed EMI (FOIR-based): ₹{max_emi:,.0f}\n"
            "- Reasons: {reasons}",
        ),
    ]
)


def build_chain(api_key: str, model: str = "gpt-4o-mini"):
    """Builds the LCEL chain: inputs dict -> prompt -> LLM -> string."""
    llm = ChatOpenAI(model=model, temperature=0.4, api_key=api_key)
    chain = RECOMMENDATION_PROMPT | llm | StrOutputParser()
    return chain


def rule_based_recommendation(inputs: dict) -> str:
    """Fallback narrative generator when no LLM/API key is available."""
    lines = []
    if inputs["eligible"] == "Yes":
        lines.append(
            f"Based on your income and existing obligations, a loan of "
            f"₹{inputs['loan_amount']:,.0f} looks affordable — your FOIR works "
            f"out to about {inputs['foir']}%, within typical bank comfort levels."
        )
    else:
        lines.append(
            f"A loan of ₹{inputs['loan_amount']:,.0f} at these terms pushes your "
            f"FOIR to about {inputs['foir']}%, above what most lenders prefer. "
            f"The FOIR-based estimate suggests up to ₹{inputs['max_loan']:,.0f} "
            f"may be a more comfortable ceiling."
        )
    lines.append("Reasons noted by the calculator: " + " ".join(inputs["reasons"]))
    lines.append(
        "Consider: reducing the loan amount, extending the tenure to lower the "
        "EMI, paying down existing EMIs first, or adding a co-applicant's income "
        "to improve eligibility."
    )
    return " ".join(lines)


def get_recommendation(calc_inputs: dict, api_key: str | None, model: str) -> str:
    prompt_vars = {
        "loan_amount": calc_inputs["loan_amount"],
        "tenure_years": calc_inputs["tenure_years"],
        "rate": calc_inputs["rate"],
        "income": calc_inputs["income"],
        "existing_emis": calc_inputs["existing_emis"],
        "credit_score": calc_inputs["credit_score"],
        "requested_emi": calc_inputs["requested_emi"],
        "foir": calc_inputs["foir"],
        "eligible": calc_inputs["eligible"],
        "max_loan": calc_inputs["max_loan"],
        "max_emi": calc_inputs["max_emi"],
        "reasons": "; ".join(calc_inputs["reasons"]),
    }

    if LANGCHAIN_AVAILABLE and api_key:
        try:
            chain = build_chain(api_key, model)
            return chain.invoke(prompt_vars)
        except Exception as e:
            return f"(LLM chain failed, showing rule-based summary instead: {e})\n\n" + rule_based_recommendation(
                prompt_vars
            )
    return rule_based_recommendation(prompt_vars)


# =============================================================================
# 3. STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Home Loan / EMI Advisor", page_icon="🏠", layout="wide")

st.title("🏠 Home Loan / EMI Advisor")
st.caption("RE-03 · Intermediate · Chain + Financial Calculator Tool")

with st.sidebar:
    st.header("Loan Details")
    loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=100000, max_value=100000000,
                                   value=4000000, step=50000)
    annual_rate = st.slider("Annual Interest Rate (%)", min_value=5.0, max_value=15.0, value=8.5, step=0.05)
    tenure_years = st.slider("Tenure (Years)", min_value=1, max_value=30, value=20)

    st.header("Applicant Profile")
    monthly_income = st.number_input("Monthly Net Income (₹)", min_value=10000, max_value=5000000,
                                      value=100000, step=5000)
    existing_emis = st.number_input("Existing Monthly EMI Obligations (₹)", min_value=0, max_value=1000000,
                                     value=5000, step=1000)
    credit_score = st.slider("Credit Score", min_value=300, max_value=900, value=740)
    foir_limit = st.slider("Bank's FOIR Limit (%)", min_value=30, max_value=65, value=50)

    st.header("AI Recommendation (optional)")
    use_llm = st.checkbox("Use LLM chain for narrative recommendation", value=False)
    api_key = None
    model_choice = "gpt-4o-mini"
    if use_llm:
        api_key = st.text_input("OpenAI API Key", type="password")
        model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        if not LANGCHAIN_AVAILABLE:
            st.warning("langchain / langchain-openai not installed — falling back to rule-based summary.")

calc = LoanCalculator()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 EMI Calculator", "✅ Eligibility Check", "💬 Recommendation", "📈 Amortization Schedule"]
)

emi = calc.calculate_emi(loan_amount, annual_rate, tenure_years)
total_payment = emi * tenure_years * 12
total_interest = total_payment - loan_amount

elig = calc.eligibility_check(
    loan_amount, monthly_income, existing_emis, annual_rate, tenure_years, credit_score, foir_limit
)

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly EMI", f"₹{emi:,.0f}")
    c2.metric("Total Interest Payable", f"₹{total_interest:,.0f}")
    c3.metric("Total Payment", f"₹{total_payment:,.0f}")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Principal", "Interest"],
                values=[loan_amount, total_interest],
                hole=0.5,
                marker=dict(colors=["#2563eb", "#f97316"]),
            )
        ]
    )
    fig.update_layout(title="Principal vs. Interest Breakup", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Eligibility Result")
    if elig["is_eligible"]:
        st.success("✅ Eligible based on FOIR and credit score thresholds.")
    else:
        st.error("❌ Not eligible at the requested amount/terms.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Requested EMI", f"₹{elig['requested_emi']:,.0f}")
    c2.metric("Current FOIR", f"{elig['current_foir_pct']}%")
    c3.metric("Max Eligible Loan", f"₹{elig['max_eligible_loan']:,.0f}")

    st.write("**Reasons:**")
    for r in elig["reasons"]:
        st.write(f"- {r}")

with tab3:
    st.subheader("Personalized Recommendation")
    calc_inputs = {
        "loan_amount": loan_amount,
        "tenure_years": tenure_years,
        "rate": annual_rate,
        "income": monthly_income,
        "existing_emis": existing_emis,
        "credit_score": credit_score,
        "requested_emi": elig["requested_emi"],
        "foir": elig["current_foir_pct"],
        "eligible": "Yes" if elig["is_eligible"] else "No",
        "max_loan": elig["max_eligible_loan"],
        "max_emi": elig["max_allowed_emi"],
        "reasons": elig["reasons"],
    }
    if st.button("Generate Recommendation"):
        with st.spinner("Running the calculator → chain pipeline..."):
            text = get_recommendation(calc_inputs, api_key if use_llm else None, model_choice)
        st.write(text)
    else:
        st.info("Click the button to run the chain (LLM-based if enabled, otherwise rule-based).")

with tab4:
    st.subheader("Full Amortization Schedule")
    schedule = calc.amortization_schedule(loan_amount, annual_rate, tenure_years)
    yearly = schedule.groupby("Year")[["Principal Paid", "Interest Paid"]].sum().reset_index()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=yearly["Year"], y=yearly["Principal Paid"], name="Principal", marker_color="#2563eb"))
    fig2.add_trace(go.Bar(x=yearly["Year"], y=yearly["Interest Paid"], name="Interest", marker_color="#f97316"))
    fig2.update_layout(barmode="stack", title="Yearly Principal vs Interest", height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(schedule, use_container_width=True, height=400)
    st.download_button(
        "Download Schedule as CSV",
        data=schedule.to_csv(index=False).encode("utf-8"),
        file_name=f"amortization_schedule_{date.today()}.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "This tool provides indicative estimates only and does not constitute a loan offer. "
    "Actual eligibility and rates are determined by the lending institution."
)
