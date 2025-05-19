# main.py
from agents.risk_analyst import RiskAnalystAgent
from agents.compliance_checker import ComplianceCheckerAgent
from tools.tot_planner import ToTPlanner

def initialize_system():
    # Initialize agents first
    agents = {
        'risk_analyst': RiskAnalystAgent(),
        'compliance_checker': ComplianceCheckerAgent() 
    }
    
    # Inject dependencies
    planner = ToTPlanner(agents=agents)
    
    # Inject planner back to agents that need it
    agents['compliance_checker'].planner = planner
    
    return planner