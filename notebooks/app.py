
import streamlit as st

# -----------------------------
# Graph Logic (formerly defense_graph.py)
# -----------------------------
class DefenseGraph:

    def invoke(self, data):
        report = data.get("report", "")

        return {
            "isr_output": self.isr_agent(report),
            "cyber_output": self.cyber_agent(report),
            "logistics_output": self.logistics_agent(report),
            "threat_output": self.threat_agent(report),
            "planner_output": self.planner_agent(report),
            "governance_output": self.governance_agent(report),
            "commander_brief": self.commander_brief(report)
        }

    def isr_agent(self, report):
        return f"ISR Analysis:\n\nIntelligence extracted from report:\n{report}"

    def cyber_agent(self, report):
        return "Cyber Assessment: No critical cyber threats detected."

    def logistics_agent(self, report):
        return "Logistics Assessment: Supply chain and assets available."

    def threat_agent(self, report):
        return "Threat Assessment: Medium risk level identified."

    def planner_agent(self, report):
        return "Mission Plan: Recommend surveillance and monitoring."

    def governance_agent(self, report):
        return "Governance Review: Mission complies with policies."

    def commander_brief(self, report):
        return (
            "Commander Summary:\n"
            "Mission intelligence reviewed. "
            "Threat level moderate. "
            "Recommend proceeding with caution."
        )


# Create graph instance
graph = DefenseGraph()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Joint Operations Center",
    layout="wide"
)

st.title("🛡️ Agentic AI Joint Operations Center")

report = st.text_area(
    "Mission Intelligence",
    height=300,
    placeholder="Enter mission intelligence report..."
)

if st.button("Run Analysis"):

    result = graph.invoke({"report": report})

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("ISR")
        st.write(result["isr_output"])

        st.subheader("Cyber")
        st.write(result["cyber_output"])

        st.subheader("Logistics")
        st.write(result["logistics_output"])

    with col2:

        st.subheader("Threat")
        st.write(result["threat_output"])

        st.subheader("Mission Planning")
        st.write(result["planner_output"])

        st.subheader("Governance")
        st.write(result["governance_output"])

    st.subheader("Commander Brief")
    st.success(result["commander_brief"])

    decision = st.radio(
        "Commander Decision",
        ["Approve", "Reject"]
    )

    st.write("Decision:", decision)
