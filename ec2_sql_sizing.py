import math
import boto3
import json
import pandas as pd
import time
import logging
import os
from functools import lru_cache
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EC2DatabaseSizingCalculator:
    # AWS instance types with updated specs
    INSTANCE_TYPES = [
        # AMD-based instances
        {"type": "m6a.large", "vCPU": 2, "RAM": 8, "max_ebs_bandwidth": 4750, "family": "general", "processor": "AMD"},
        {"type": "m6a.xlarge", "vCPU": 4, "RAM": 16, "max_ebs_bandwidth": 9500, "family": "general", "processor": "AMD"},
        {"type": "m6a.2xlarge", "vCPU": 8, "RAM": 32, "max_ebs_bandwidth": 19000, "family": "general", "processor": "AMD"},
        {"type": "m6a.4xlarge", "vCPU": 16, "RAM": 64, "max_ebs_bandwidth": 38000, "family": "general", "processor": "AMD"},
        {"type": "m6a.8xlarge", "vCPU": 32, "RAM": 128, "max_ebs_bandwidth": 47500, "family": "general", "processor": "AMD"},
        {"type": "r6a.large", "vCPU": 2, "RAM": 16, "max_ebs_bandwidth": 4750, "family": "memory", "processor": "AMD"},
        {"type": "r6a.xlarge", "vCPU": 4, "RAM": 32, "max_ebs_bandwidth": 9500, "family": "memory", "processor": "AMD"},
        {"type": "r6a.2xlarge", "vCPU": 8, "RAM": 64, "max_ebs_bandwidth": 19000, "family": "memory", "processor": "AMD"},
        {"type": "r6a.4xlarge", "vCPU": 16, "RAM": 128, "max_ebs_bandwidth": 38000, "family": "memory", "processor": "AMD"},
        {"type": "r6a.8xlarge", "vCPU": 32, "RAM": 256, "max_ebs_bandwidth": 47500, "family": "memory", "processor": "AMD"},
        {"type": "c6a.large", "vCPU": 2, "RAM": 4, "max_ebs_bandwidth": 4750, "family": "compute", "processor": "AMD"},
        {"type": "c6a.xlarge", "vCPU": 4, "RAM": 8, "max_ebs_bandwidth": 9500, "family": "compute", "processor": "AMD"},
        {"type": "c6a.2xlarge", "vCPU": 8, "RAM": 16, "max_ebs_bandwidth": 19000, "family": "compute", "processor": "AMD"},
        {"type": "c6a.4xlarge", "vCPU": 16, "RAM": 32, "max_ebs_bandwidth": 38000, "family": "compute", "processor": "AMD"},
        {"type": "c6a.8xlarge", "vCPU": 32, "RAM": 64, "max_ebs_bandwidth": 47500, "family": "compute", "processor": "AMD"},
        
        # Intel-based instances
        {"type": "m6i.large", "vCPU": 2, "RAM": 8, "max_ebs_bandwidth": 4750, "family": "general", "processor": "Intel"},
        {"type": "m6i.xlarge", "vCPU": 4, "RAM": 16, "max_ebs_bandwidth": 9500, "family": "general", "processor": "Intel"},
        {"type": "m6i.2xlarge", "vCPU": 8, "RAM": 32, "max_ebs_bandwidth": 19000, "family": "general", "processor": "Intel"},
        {"type": "m6i.4xlarge", "vCPU": 16, "RAM": 64, "max_ebs_bandwidth": 38000, "family": "general", "processor": "Intel"},
        {"type": "m6i.8xlarge", "vCPU": 32, "RAM": 128, "max_ebs_bandwidth": 47500, "family": "general", "processor": "Intel"},
        {"type": "r6i.large", "vCPU": 2, "RAM": 16, "max_ebs_bandwidth": 4750, "family": "memory", "processor": "Intel"},
        {"type": "r6i.xlarge", "vCPU": 4, "RAM": 32, "max_ebs_bandwidth": 9500, "family": "memory", "processor": "Intel"},
        {"type": "r6i.2xlarge", "vCPU": 8, "RAM": 64, "max_ebs_bandwidth": 19000, "family": "memory", "processor": "Intel"},
        {"type": "r6i.4xlarge", "vCPU": 16, "RAM": 128, "max_ebs_bandwidth": 38000, "family": "memory", "processor": "Intel"},
        {"type": "r6i.8xlarge", "vCPU": 32, "RAM": 256, "max_ebs_bandwidth": 47500, "family": "memory", "processor": "Intel"},
        {"type": "c6i.large", "vCPU": 2, "RAM": 4, "max_ebs_bandwidth": 4750, "family": "compute", "processor": "Intel"},
        {"type": "c6i.xlarge", "vCPU": 4, "RAM": 8, "max_ebs_bandwidth": 9500, "family": "compute", "processor": "Intel"},
        {"type": "c6i.2xlarge", "vCPU": 8, "RAM": 16, "max_ebs_bandwidth": 19000, "family": "compute", "processor": "Intel"},
        {"type": "c6i.4xlarge", "vCPU": 16, "RAM": 32, "max_ebs_bandwidth": 38000, "family": "compute", "processor": "Intel"},
        {"type": "c6i.8xlarge", "vCPU": 32, "RAM": 64, "max_ebs_bandwidth": 47500, "family": "compute", "processor": "Intel"},
    ]
    
    # Environment multipliers
    ENV_MULTIPLIERS = {
        "PROD": {"cpu_ram": 1.0, "storage": 1.0},
        "SQA": {"cpu_ram": 0.75, "storage": 0.7},
        "QA": {"cpu_ram": 0.6, "storage": 0.5},
        "DEV": {"cpu_ram": 0.4, "storage": 0.3}
    }
    
    # Base pricing data (validated against AWS as of 2023-10)
    BASE_PRICING = {
        "instance": {
            # AMD-based instances (Windows pricing)
            "m6a.large": 0.236,
            "m6a.xlarge": 0.472,
            "m6a.2xlarge": 0.944,
            "m6a.4xlarge": 1.888,
            "m6a.8xlarge": 3.776,
            "r6a.large": 0.278,
            "r6a.xlarge": 0.556,
            "r6a.2xlarge": 1.112,
            "r6a.4xlarge": 2.224,
            "r6a.8xlarge": 4.448,
            "c6a.large": 0.215,
            "c6a.xlarge": 0.430,
            "c6a.2xlarge": 0.860,
            "c6a.4xlarge": 1.720,
            "c6a.8xlarge": 3.440,
            
            # Intel-based instances (Windows pricing)
            "m6i.large": 0.272,
            "m6i.xlarge": 0.544,
            "m6i.2xlarge": 1.088,
            "m6i.4xlarge": 2.176,
            "m6i.8xlarge": 4.352,
            "r6i.large": 0.318,
            "r6i.xlarge": 0.636,
            "r6i.2xlarge": 1.272,
            "r6i.4xlarge": 2.544,
            "r6i.8xlarge": 5.088,
            "c6i.large": 0.248,
            "c6i.xlarge": 0.496,
            "c6i.2xlarge": 0.992,
            "c6i.4xlarge": 1.984,
            "c6i.8xlarge": 3.968,
        },
        "ebs": {
            "us-east-1": {
                "gp3": {"gb": 0.08, "iops": 0.005, "throughput": 0.04},
                "io2": {"gb": 0.125, "iops": 0.065}
            },
            "us-west-1": {
                "gp3": {"gb": 0.088, "iops": 0.0055, "throughput": 0.044},
                "io2": {"gb": 0.138, "iops": 0.0715}
            },
            "us-west-2": {
                "gp3": {"gb": 0.084, "iops": 0.0052, "throughput": 0.042},
                "io2": {"gb": 0.131, "iops": 0.068}
            }
        }
    }
    
    # Cache for pricing data (region -> instance_type -> price)
    PRICING_CACHE = {}
    CACHE_EXPIRY = 24 * 3600  # 24 hours
    
    def __init__(self):
        self.inputs = {
            "region": "us-east-1",
            "on_prem_cores": 16,
            "peak_cpu_percent": 65,
            "on_prem_ram_gb": 64,
            "peak_ram_percent": 75,
            "storage_current_gb": 500,
            "storage_growth_rate": 0.15,
            "peak_iops": 8000,
            "peak_throughput_mbps": 400,
            "years": 3,
            "workload_profile": "general",
            "prefer_amd": True
        }
        self.instance_pricing = self.BASE_PRICING["instance"].copy()
        self.ebs_pricing = self.BASE_PRICING["ebs"].copy()
        self.last_fetch_time = 0
        self.recommendation_cache = {}
    
    def validate_aws_credentials(self, region=None):
        """Check if valid AWS credentials are available with detailed diagnostics"""
        try:
            # Use provided region or default to us-east-1
            region = region or self.inputs.get("region", "us-east-1")
            sts = boto3.client('sts', region_name=region)
            identity = sts.get_caller_identity()
            
            # Get account ID from ARN
            account_id = identity['Arn'].split(':')[4]
            return True, f"Valid AWS credentials (Account: {account_id}, Region: {region})"
        
        except NoCredentialsError:
            return False, "No AWS credentials found in environment, config, or IAM role"
            
        except PartialCredentialsError as e:
            return False, f"Partial credentials: {str(e)}"
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            
            if error_code == "InvalidClientTokenId":
                return False, "Invalid Access Key ID"
            elif error_code == "SignatureDoesNotMatch":
                return False, "Invalid Secret Access Key"
            elif error_code == "AccessDenied":
                return False, "Access denied - check IAM permissions"
            else:
                return False, f"AWS API error: {error_code} - {error_msg}"
                
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"
    
    def fetch_current_prices(self, force_refresh=False):
        """Fetch current prices with caching and validation"""
        region = self.inputs["region"]
        
        # Use cached prices if available and not expired
        current_time = time.time()
        if not force_refresh and region in self.PRICING_CACHE:
            cache_time, cached_prices = self.PRICING_CACHE[region]
            if current_time - cache_time < self.CACHE_EXPIRY:
                self.instance_pricing.update(cached_prices)
                logger.info(f"Using cached prices for {region}")
                return
        
        # Validate credentials
        cred_status, cred_message = self.validate_aws_credentials()
        if not cred_status:
            logger.warning(f"AWS credentials not available: {cred_message}. Using base pricing.")
            return
        
        try:
            # Initialize pricing client
            pricing_client = boto3.client('pricing', region_name='us-east-1')
            
            # Build filters
            filters = [
                {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Windows'},
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region},
                {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
                {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                {'Type': 'TERM_MATCH', 'Field': 'licenseModel', 'Value': 'License Included'},
                {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
            ]
            
            updated_prices = {}
            for instance_type in self.BASE_PRICING["instance"].keys():
                try:
                    # Query AWS Pricing API
                    response = pricing_client.get_products(
                        ServiceCode='AmazonEC2',
                        Filters=filters + [
                            {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type}
                        ],
                        MaxResults=1
                    )
                    
                    if response['PriceList']:
                        price_item = json.loads(response['PriceList'][0])
                        terms = price_item['terms']['OnDemand']
                        price_dimensions = list(terms.values())[0]['priceDimensions']
                        price = list(price_dimensions.values())[0]['pricePerUnit']['USD']
                        
                        # Validate price difference
                        base_price = self.BASE_PRICING["instance"].get(instance_type, 0)
                        if base_price > 0:
                            diff_percent = abs(float(price) - base_price) / base_price * 100
                            if diff_percent > 10:
                                logger.warning(
                                    f"Significant price difference for {instance_type}: "
                                    f"Base ${base_price:.3f} vs Current ${float(price):.3f} "
                                    f"({diff_percent:.1f}%)"
                                )
                        
                        updated_prices[instance_type] = float(price)
                except Exception as e:
                    logger.error(f"Error fetching price for {instance_type}: {str(e)}")
                    continue
            
            # Update cache and instance pricing
            self.PRICING_CACHE[region] = (current_time, updated_prices)
            self.instance_pricing.update(updated_prices)
            logger.info(f"Fetched {len(updated_prices)} updated prices for {region}")
            
        except Exception as e:
            logger.error(f"Pricing API error: {str(e)}")
    
    def calculate_requirements(self, env):
        """Calculate requirements with optimized math operations"""
        mult = self.ENV_MULTIPLIERS[env]
        
        # Calculate compute requirements
        vcpus = max(
            math.ceil(
                self.inputs["on_prem_cores"] * 
                (self.inputs["peak_cpu_percent"] / 100) *
                1.2 / 0.7 *  # Buffer and efficiency factor
                mult["cpu_ram"]
            ), 
            2  # Minimum
        )
        
        ram = max(
            math.ceil(
                self.inputs["on_prem_ram_gb"] * 
                (self.inputs["peak_ram_percent"] / 100) *
                1.2 / 0.7 *  # Buffer and efficiency factor
                mult["cpu_ram"]
            ),
            4  # Minimum
        )
        
        # Calculate storage requirements
        growth_factor = (1 + self.inputs["storage_growth_rate"]) ** self.inputs["years"]
        storage = max(
            math.ceil(
                self.inputs["storage_current_gb"] * 
                growth_factor * 
                1.3 *  # Buffer
                mult["storage"]
            ),
            20  # Minimum
        )
        
        # Calculate I/O requirements
        iops_required = math.ceil(self.inputs["peak_iops"] * 1.3)
        throughput_required = math.ceil(self.inputs["peak_throughput_mbps"] * 1.3)
        
        # Determine EBS type
        ebs_type = "io2" if (iops_required > 16000 or throughput_required > 1000) else "gp3"
        
        # Select instance
        instance = self.select_instance(
            vcpus, ram, throughput_required, 
            self.inputs["workload_profile"], 
            self.inputs["prefer_amd"]
        )
        
        # Calculate costs
        instance_cost = self.calculate_instance_cost(instance["type"])
        ebs_cost = self.calculate_ebs_cost(
            ebs_type, 
            storage, 
            iops_required, 
            throughput_required
        )
        total_cost = instance_cost + ebs_cost
        
        return {
            "instance_type": instance["type"],
            "vCPUs": vcpus,
            "RAM_GB": ram,
            "storage_GB": storage,
            "ebs_type": ebs_type,
            "iops_required": iops_required,
            "throughput_required": f"{throughput_required} MB/s",
            "family": instance["family"],
            "processor": instance["processor"],
            "instance_cost": instance_cost,
            "ebs_cost": ebs_cost,
            "total_cost": total_cost
        }
    
    def select_instance(self, required_vcpus, required_ram, required_throughput, workload_profile, prefer_amd):
        """Optimized instance selection with pre-filtering"""
        # Pre-filter instances
        candidates = [
            i for i in self.INSTANCE_TYPES 
            if (prefer_amd or i["processor"] != "AMD") and
               (workload_profile == "general" or i["family"] == workload_profile) and
               i["vCPU"] >= required_vcpus and
               i["RAM"] >= required_ram and
               i["max_ebs_bandwidth"] >= (required_throughput * 1.2)
        ]
        
        # If no candidates found, return largest available
        if not candidates:
            return max(
                [i for i in self.INSTANCE_TYPES if prefer_amd or i["processor"] != "AMD"],
                key=lambda x: x["vCPU"]
            )
        
        # Prioritize AMD for cost savings if enabled
        if prefer_amd:
            amd_candidates = [i for i in candidates if i["processor"] == "AMD"]
            if amd_candidates:
                return min(amd_candidates, key=lambda x: (x["vCPU"], x["RAM"]))
        
        # Return smallest suitable instance
        return min(candidates, key=lambda x: (x["vCPU"], x["RAM"]))
    
    def calculate_instance_cost(self, instance_type):
        """Calculate monthly instance cost"""
        hourly_rate = self.instance_pricing.get(instance_type, 0)
        return round(hourly_rate * 24 * 30, 2)  # Monthly cost
    
    def calculate_ebs_cost(self, ebs_type, storage_gb, iops, throughput_mbps):
        """Calculate monthly EBS storage cost"""
        region = self.inputs["region"]
        
        # Get pricing for region
        region_pricing = self.ebs_pricing.get(region, self.ebs_pricing["us-east-1"])
        pricing = region_pricing.get(ebs_type)
        
        if not pricing:
            return 0.0
        
        base_cost = storage_gb * pricing["gb"]
        
        if ebs_type == "gp3":
            # Extra cost for provisioned IOPS above 3000
            extra_iops = max(0, iops - 3000)
            iops_cost = extra_iops * pricing["iops"]
            
            # Extra cost for provisioned throughput above 125 MB/s
            extra_throughput = max(0, throughput_mbps - 125)
            throughput_cost = extra_throughput * pricing["throughput"]
            
            return round(base_cost + iops_cost + throughput_cost, 2)
        
        elif ebs_type == "io2":
            # io2 charges per provisioned IOPS
            iops_cost = iops * pricing["iops"]
            return round(base_cost + iops_cost, 2)
        
        return round(base_cost, 2)
    
    def generate_all_recommendations(self):
        """Generate recommendations with input-based caching"""
        # Create cache key based on inputs
        cache_key = hash(frozenset(self.inputs.items()))
        
        # Return cached results if available
        if cache_key in self.recommendation_cache:
            logger.info("Using cached recommendations")
            return self.recommendation_cache[cache_key]
        
        # Generate new recommendations
        results = {}
        for env in self.ENV_MULTIPLIERS.keys():
            results[env] = self.calculate_requirements(env)
        
        # Cache results
        self.recommendation_cache[cache_key] = results
        return results