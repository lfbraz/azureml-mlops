# Train Model
az ml job create -f jobs/train.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv

# Register Model
az ml job create -f jobs/register.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv

# Deploy Dev
az ml job create -f jobs/deploy-dev.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv

# Deploy Prod
az ml job create -f jobs/deploy-prod.yml --resource-group RG-ML --workspace-name aml-labs --query name -o tsv