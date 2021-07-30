while True:

    init_col = float(input("Initial collateral: "))
    util_rate = float(input("  Utilization rate: "))
    safety_margin = float(input("     Safety margin: "))
    print()

    coll = init_col
    tmp = init_col
    bor = 0
    iteration = 1

    lim = init_col/(1-(util_rate*(1-safety_margin)))
    sumlim = lim * (1 + (util_rate * (1 - safety_margin)))

    while 1 - ( (bor+coll) / sumlim) > 0.05:

        print("Iteration", iteration)
        iteration += 1

        # borrow
        itmp = tmp
        tmp = tmp * util_rate * (1-safety_margin)
        coll += tmp
        bor += tmp

        print(f'Borrow {util_rate * (1 - safety_margin) * 100:.2f} % '
              f'of {itmp:.2f} = {tmp:.2f} ({util_rate*itmp:.2f} available) and lend it again\n'
              f'=> Collateral = {coll:.2f}, Borrowed = {bor:.2f}\n')

    print(f'Collateral: {coll:7.2f}\n'
          f'Borrowed:   {bor:7.2f}\n'
          f'Sum:        {coll+bor:7.2f}\n'
          f'Limit:      {sumlim:7.2f}\n\n'
          f'Bor/Col:    {bor/coll*100:7.2f} %\n'
          f'Health:    {coll/bor:7.1f}\n')