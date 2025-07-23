def euler_update(pot_Helfand_i, pot_Flory_plus_i, w_minus, Rho_A, Rho_B, Rho0, Kappa, Chi, N, dt_helfand, dt_plus, dt_minus):
    #Use the equations to update potential fields with euler method
    delta_w_helfand_i = -(((Rho0 * (pot_Helfand_i)/(Kappa*(N)**2)) + (Rho_B + Rho_A - Rho0)/(N))) * dt_helfand

    delta_w_flory_i = -((2 * Rho0 * (pot_Flory_plus_i) /(Chi*(N)**2)) + (Rho_B + Rho_A)/(N)) * dt_plus
    
    delta_w_minus = -((2 * Rho0 * w_minus / (Chi*(N)**2)) + (Rho_B - Rho_A)/(N)) * dt_minus 

    pot_Helfand_i = pot_Helfand_i + delta_w_helfand_i
    pot_Flory_plus_i = pot_Flory_plus_i + delta_w_flory_i
    w_minus = w_minus + delta_w_minus

    return pot_Helfand_i, pot_Flory_plus_i, w_minus
