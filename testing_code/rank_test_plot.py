import matplotlib.pyplot as plt

# Original experiment results
results = """
Rank = 4
emailservice,0.42055731,2
currencyservice,0.43637273,1
Rank = 3
emailservice,0.42055731,2
currencyservice,0.43637273,1
Rank = 2
emailservice,0.40091858,2
currencyservice,0.43637273,1
Rank = 1
emailservice,0.30023167,2
currencyservice,0.41124949,1
"""

# Parse experiment results
ranks = []
email_service_results = []
currency_service_results = []

lines = results.strip().split('\n')
for i in range(0, len(lines), 3):
    rank = int(lines[i].split('=')[1].strip())
    ranks.append(rank)

    email_service_result = float(lines[i + 1].split(',')[1])
    email_service_results.append(email_service_result)

    currency_service_result = float(lines[i + 2].split(',')[1])
    currency_service_results.append(currency_service_result)

# Plot the graph
plt.plot(ranks, email_service_results, marker='o', label='emailservice')
plt.plot(ranks, currency_service_results, marker='o', label='currencyservice')

# Add labels and title
plt.xlabel('Rank')
plt.ylabel('MI')
# plt.title('Impact of Rank on Email Service and Currency Service')

# Add a legend
plt.legend()

# Display the graph
plt.show()
