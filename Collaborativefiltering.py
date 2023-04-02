import numpy as np

class CollaborativeFiltering:
    def __init__(self, num_users, num_items, k=10):
        self.num_users = num_users
        self.num_items = num_items
        self.k = k
        self.user_factors = np.random.normal(size=(num_users, k))
        self.item_factors = np.random.normal(size=(num_items, k))

    def fit(self, ratings, epochs=10, lr=0.01, reg=0.1):
        for epoch in range(epochs):
            for user, item, rating in ratings:
                error = rating - np.dot(self.user_factors[user], self.item_factors[item])
                user_factor_gradient = -2 * error * self.item_factors[item] + 2 * reg * self.user_factors[user]
                item_factor_gradient = -2 * error * self.user_factors[user] + 2 * reg * self.item_factors[item]
                self.user_factors[user] -= lr * user_factor_gradient
                self.item_factors[item] -= lr * item_factor_gradient

    def predict(self, user, item):
        return np.dot(self.user_factors[user], self.item_factors[item])

if __name__ == '__main__':
    ratings = [(0, 0, 5), (0, 1, 3), (1, 0, 4), (1, 1, 2), (2, 0, 1), (2, 1, 4), (2, 2, 5)]
    cf = CollaborativeFiltering(3, 3, k=2)
    cf.fit(ratings)
    print(cf.predict(0, 2))
