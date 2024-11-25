import pandas as pd
import random
from faker import Faker


class CustomerGenerator:
    def __init__(self, age_distribution):
        self.age_distribution = self._normalize_age_distribution(age_distribution)
        self.fake = Faker()

    def _normalize_age_distribution(self, age_distribution):

        """Normalize the age distribution probabilities to sum to 100%"""
        
        total_prob = sum(prob for _, _, prob in age_distribution)
        return [(start, end, (prob / total_prob) * 100) for start, end, prob in age_distribution]

    def _generate_age(self):

        """Generate a random age based on the normalized distribution"""

        rand_value = random.uniform(0, 100)
        cumulative_probability = 0

        for start_age, end_age, probability in self.age_distribution:
            cumulative_probability += probability
            if rand_value <= cumulative_probability:
                age = random.randint(start_age, end_age)
                return max(18, min(age, 65))

    def generate_customer(self):

        """Generate a new customer with random attributes"""

        return {
            "customer_id": self.fake.uuid4(),
            "gender": random.choice(["Male", "Female"]),
            "age": self._generate_age()
        }


class TransactionGenerator:

    def __init__(self, preferences):

        self.preferences = preferences

    def _get_generation(self, age):

        """Determine generation based on age"""

        for gen_name, gen_info in self.preferences.items():
            if gen_info["Age Range"][0] <= age <= gen_info["Age Range"][1]:
                return gen_name
        raise ValueError(f"Age {age} does not fit into any defined generation in preferences.")

    def _get_product_category(self, generation, gender):

        """Select a product category based on generational and gender preferences"""

        gen_preferences = self.preferences[generation]["Shopping Preference"][gender]
        return random.choices(
            population=list(gen_preferences.keys()),
            weights=[prefs[0] for prefs in gen_preferences.values()],
            k=1
        )[0]

    def _calculate_adjusted_amount(self, base_amount, generation, gender, product_category):

        """Calculate adjusted transaction amount based on preferences"""

        spending_deviation = self.preferences[generation]["Shopping Preference"][gender][product_category][1]
        return round(base_amount * random.uniform(1 - spending_deviation, 1 + spending_deviation))

    def generate_transaction(self, customer, base_amount, campaign_id, date, transaction_id):

        """Generate a new transaction for a customer"""

        generation = self._get_generation(customer["age"])
        product_category = self._get_product_category(generation, customer["gender"])
        adjusted_amount = self._calculate_adjusted_amount(base_amount, generation, customer["gender"], product_category)

        return {
            "campaign_id": campaign_id,
            "transaction_id": transaction_id,
            "amount": adjusted_amount,
            "transaction_date": date,
            "customer_id": customer["customer_id"],
            "gender": customer["gender"],
            "age": customer["age"],
            "product_category": product_category,
        }


class CampaignDataset:
    def __init__(self, df_campaigns: pd.DataFrame, df_metrics: pd.DataFrame, df_transactions: pd.DataFrame):
        self.campaigns = df_campaigns
        self.metrics = df_metrics
        self.transactions = df_transactions
        self.daily_metrics = None  # To store simulated daily metrics after processing
        self.customer_pool = []

    def _merge_campaigns_and_metrics(self):
        
        """Merge campaign data with metrics data based on campaign_id"""

        merged = pd.merge(self.campaigns, self.metrics, on="campaign_id", how="inner")
        if merged.empty:
            raise ValueError("No matching campaign data found in metrics.")
        return merged
    
    def _get_daily_weight(self, date, dow_weights):

        """
        Calculate weights multiplier for a given date based on day of week
        and typical daily patterns.
        """
        
        # Get base weight for day of week
        day_of_week = date.weekday()
        base_weight = dow_weights[day_of_week]
            
        return base_weight

    def simulate_daily_metrics(self, campaign_id, dow_weights):

        """Simulate daily metrics for a campaign."""

        merged_data = self._merge_campaigns_and_metrics()
        campaign = merged_data[merged_data["campaign_id"] == campaign_id].iloc[0]
        start_date, end_date = campaign["start_date"], campaign["end_date"]
        total_days = (end_date - start_date).days + 1

        total_impressions = campaign["Impressions"]
        total_clicks = campaign["Clicks"]
        total_web_hits = campaign["website_landing_hits"]
        base_ctr = total_clicks/total_impressions

        # Get the weights for each day of the campaign
        daily_weights = [self._get_daily_weight(start_date + pd.Timedelta(days=i), dow_weights) 
                        for i in range(total_days)]
        total_weight = sum(daily_weights)

        # Normalize weights and calculate base daily impressions and web hits
        daily_weights = [w/total_weight for w in daily_weights]
        base_daily_impressions = [int(total_impressions * w) for w in daily_weights]
        base_daily_web_hits = [int(total_web_hits * w) for w in daily_weights]
        
        
        # Add random variation while preserving the pattern
        variation_factor = 0.02  # 2% maximum variation

        daily_impressions = [
            int(imp + (imp * random.gauss(0, variation_factor)))
            for imp in base_daily_impressions
        ]


        daily_clicks = [
            int(impression * (base_ctr + random.gauss(0, variation_factor)))
            for impression in daily_impressions
        ]

        daily_website_landing_hits = [int(hits) + (hits * random.gauss(0,variation_factor))
                                      for hits in base_daily_web_hits]

        # Scale clicks and impressions to sum up to the total campaign metrics
        scale_factor = total_clicks / sum(daily_clicks)
        daily_clicks = [int(click * scale_factor) for click in daily_clicks]
            
        scale_factor = total_impressions / sum(daily_impressions)
        daily_impressions = [int(imp * scale_factor) for imp in daily_impressions]

        scale_factor = total_web_hits / sum(daily_website_landing_hits)
        daily_website_landing_hits = [int(hits * scale_factor) for hits in base_daily_web_hits]

        # Add possible deficits to the maximum value so that the CTR is not affected
        impressions_deficit = total_impressions - sum(daily_impressions)
        clicks_deficit = total_clicks - sum(daily_clicks)
        web_hits_deficit = total_web_hits - sum(daily_website_landing_hits)

        max_impressions_index = daily_impressions.index(max(daily_impressions))
        daily_impressions[max_impressions_index] = daily_impressions[max_impressions_index] + impressions_deficit
        daily_clicks[max_impressions_index] = daily_clicks[max_impressions_index] + clicks_deficit
        daily_website_landing_hits[max_impressions_index] = daily_website_landing_hits[max_impressions_index] + web_hits_deficit

        daily_data = pd.DataFrame({
            "campaign_id": campaign_id,
            "date": [start_date + pd.Timedelta(days=i) for i in range(total_days)],
            "daily_web_hits": daily_website_landing_hits,
            "daily_impressions": daily_impressions,
            "daily_clicks": daily_clicks,
        })

        self.daily_metrics = daily_data
        return daily_data

    def generate_synthetic_transactions(self, campaign_id: int, preferences: dict, age_distribution: list, click_conversion: float):
        """Generate synthetic transactions for a campaign based on daily metrics with 10% chance of customer duplication."""

        if self.daily_metrics is None or self.daily_metrics.empty:
            raise ValueError("Daily metrics not generated. Please run simulate_daily_metrics first.")

        daily_data = self.daily_metrics[self.daily_metrics["campaign_id"] == campaign_id]
        if daily_data.empty:
            raise ValueError(f"No daily metrics found for campaign_id {campaign_id}")

        customer_gen = CustomerGenerator(age_distribution)
        transaction_gen = TransactionGenerator(preferences)
        new_rows = []
        next_transaction_id = int(self.transactions["transaction_id"].max() + 1) if not self.transactions.empty else 1

        variation = 0.01
        for _, row in daily_data.iterrows():
            num_transactions = int(row["daily_clicks"] * (click_conversion + random.gauss(0, variation))) 
            for _ in range(num_transactions):
                # 10% chance to reuse an existing customer if there are any in the pool
                if self.customer_pool and random.random() < 0.10:
                    customer = random.choice(self.customer_pool)
                else:
                    customer = customer_gen.generate_customer()
                    self.customer_pool.append(customer)  # Add new customer to the pool

                new_transaction = transaction_gen.generate_transaction(
                    customer=customer,
                    base_amount=random.randint(75, 500),  # Randomize base amount
                    campaign_id=campaign_id,
                    date=row["date"],
                    transaction_id=next_transaction_id + len(new_rows),
                )
                new_rows.append(new_transaction)

        return pd.concat([self.transactions[self.transactions['campaign_id'] == campaign_id], pd.DataFrame(new_rows)], ignore_index=True)