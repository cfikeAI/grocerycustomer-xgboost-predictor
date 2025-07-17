terraform {
    required_providers {
      azurerm = {
        source = "hashicorp/azurerm"
        version = "~> 3.0"
      }
    }
}

provider "azurerm" {
    features {}
}

#Resource Group
resource "azurerm_resource_group" "rg" {
    name = var.resource_group_name
    location = var.location
}

#Azure Container Registry
resource "azurerm_container_registry" "acr" {
    name = var.acr_name
    resource_group_name = azurerm_resource_group.rg.name
    location = azurerm_resource_group.rg.location
    sku = "Basic"
    admin_enabled = true
}

#AKS Cluster
resource "azurerm_kubernetes_cluster" "aks" {
    name = var.aks_name
    location = var.azurerm_resource_group.location
    resource_group_name = var.azurerm_resource_group.name
    dns_prefix = "gcp-aks"

    default_node_pool {
      name = "default"
      node_count = 2
      vm_size = "Standard_DS2_v2"
    }

    identity {
      type = "SystemAssigned"
    }
}


#Give AKS permission to pull from ACR
resource "azurerm_role_assignment" "acr-pull" {
    principal_id = azurerm_kubernetes_cluster.aks.kubelet_identity[0].object_id
    role_definition_name = "ACRPull"
    scope = azurerm_container_registry.acr.id
}