# Submit job and get the run_id
run_id=$(az ml job create -f jobs/train.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv)
#current_state=$(az ml job show -n "f14aa6e2-dcf4-4528-8c9d-bbf2a01b8136" --resource-group RG-ML --workspace-name aml-labs --query "{Jobstatus:status}")
#current_state=$(echo $current_state | jq '.Jobstatus' |  tr -d '"')

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