from math import log

loan = float(input("            Loan: "))
loan_credit = (float(input("Loan credit in %: "))/100)
daily_yield = 1 + (float(input("Daily yield in %: "))/100)

tax_rate = 0.25

# (loan * yield**d - loan) * (1-tax_rate) = loan * (1 + loan_credit) - loan
payoff_days = log(((loan * (1+loan_credit)) / (1-tax_rate) + loan) / loan, daily_yield)


def get_profit_tax(d):
    gain = loan * daily_yield**d - loan
    tax = gain * tax_rate
    profit = gain-tax
    return profit, tax


p, t = get_profit_tax(31)
one_m_profit = p
one_m_tax = t
one_m_gain = p+t

p, t = get_profit_tax(180)
six_m_profit = p
six_m_tax = t
six_m_gain = p+t

p, t = get_profit_tax(365)
one_y_profit = p
one_y_tax = t
one_y_gain = p+t

print(
    f'\n'
    f'Payoff after: {int(payoff_days+1):d}\n\n'
    f'Profit after 1 month: {one_m_profit:10.2f}\t\tGain: {one_m_gain:10.2f}\t\tTax: {one_m_tax:10.2f}\n'
    f'Profit after 6 month: {six_m_profit:10.2f}\t\tGain: {six_m_gain:10.2f}\t\tTax: {six_m_tax:10.2f}\n'
    f' Profit after 1 year: {one_y_profit:10.2f}\t\tGain: {one_y_gain:10.2f}\t\tTax: {one_y_tax:10.2f}\n'
)