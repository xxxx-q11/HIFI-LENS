root="./"
apikeys={}
apikeys['deepseek_api_key']=""
apikeys['think_model_name']="deepseek-reasoner"
#apikeys['think_model_name']="deepseek-chat"
apikeys['generate_model_name']="deepseek-chat"
base_agentconfig={}
base_agentconfig["start_date"] = "2025-03-01"
base_agentconfig["end_date"] = "2025-03-03"
base_agentconfig['company'] = "JNJ"
base_agentconfig['event'] = "Trump"
base_config={
    'apikeys':apikeys,
    'root':root,
    'agent_config':base_agentconfig
}
