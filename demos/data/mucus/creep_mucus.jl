using CSV
using DataFrames
using PyPlot
using RHEOS

data_original = importcsv("creep_mucus_data_julia.csv", t_col = 1, ϵ_col = 2, σ_col = 3)
data = data_original
rheotimedatatype(data)

# Fit model
# Lower bounds
Lo = (η = 1e-2, cᵦ=1e-2, β = 0.001)
# Upper bounds
Hi = (η = 1e2, cᵦ=1e2, β = 0.99)

# Initial parameters
P0 = (η = 1e0, cᵦ=1e-1, β = 0.5)


# Time the fitting process
@time begin
    fractDamperKV = modelfit(data, FractD_KelvinVoigt, stress_imposed, lo = Lo, hi = Hi, p0 = P0)
end

# Predict data
data_ext = extract(data_original, 1)

rheotimedatatype(data_ext)

fractDamperKV_predict = modelpredict(data_ext, fractDamperKV)

# Exporting data
#CSV.write("export_timedata_mucus_original_0.csv", DataFrame(time=data.t))
#CSV.write("export_straindata_mucus_original_0.csv", DataFrame(strain=data.ϵ))

CSV.write("export_timedata_mucus_0.csv", DataFrame(time=fractDamperKV_predict.t))
CSV.write("export_straindata_mucus_0.csv", DataFrame(strain=fractDamperKV_predict.ϵ))

# Calculate Mean Absolute Percentage Error (MAPE)
# Assuming Fzener_model_predict.σ contains the predicted values
rss = sum(((data.ϵ .- fractDamperKV_predict.ϵ) ./ data.ϵ)**2)  # Calculate MAPE
println("rss: ", mape)

# Now we can plot data and model together for comparison
fig, ax = subplots(1, 1, figsize = (7, 5))
ax.plot(data.t, data.ϵ, "o", markersize = 5, label="Experimental Data")  # Original data
ax.plot(fractDamperKV_predict.t, fractDamperKV_predict.ϵ, color="red", label="Fitted Model")  # Fitted model
ax.set_xlabel("Time")
ax.set_ylabel("Strain")
ax.legend()
fig.savefig("mucus_rheos.png")

