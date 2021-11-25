# Submit job and get the run_id
run_id=$(az ml job create -f jobs/deploy-dev.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv)

# Check current state
current_state=""
while [ "$current_state" != "Completed" ]
do
    current_state=$(az ml job show -n $run_id --resource-group RG-ML --workspace-name aml-labs --query "{Jobstatus:status}")
    current_state=$(echo $current_state | jq '.Jobstatus' |  tr -d '"')

    sleep 5
done

echo "Finishing in state: $current_state"
exit 0 # Success