#!/bin/bash
output=$(az ml workspace create --resource-group RG-ML --name aml-labs --only-show-errors 2>&1)

if [[ "$output" == *"ERROR"* ]]; then
    output=$(az ml workspace show --resource-group RG-ML --name aml-labs --only-show-errors)
    echo "Workspace Encontrada: $output"
fi