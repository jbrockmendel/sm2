__all__ = ["gls", "wls", "ols", "glsar",
           "glm",
           "rlm",
           "mnlogit", "logit", "probit", "poisson", "negativebinomial"]
import sm2.discrete.discrete_model as dm_
# import sm2.duration.hazard_regression as hr_
import sm2.genmod.generalized_linear_model as glm_
# import sm2.genmod.generalized_estimating_equations as gee_
import sm2.regression.linear_model as lm_
# import sm2.regression.mixed_linear_model as mlm_
# import sm2.regression.quantile_regression as qr_
import sm2.robust.robust_linear_model as roblm_

gls = lm_.GLS.from_formula
wls = lm_.WLS.from_formula
ols = lm_.OLS.from_formula
glsar = lm_.GLSAR.from_formula
# mixedlm = mlm_.MixedLM.from_formula
glm = glm_.GLM.from_formula
rlm = roblm_.RLM.from_formula
mnlogit = dm_.MNLogit.from_formula
logit = dm_.Logit.from_formula
probit = dm_.Probit.from_formula
poisson = dm_.Poisson.from_formula
negativebinomial = dm_.NegativeBinomial.from_formula
# quantreg = qr_.QuantReg.from_formula
# phreg = hr_.PHReg.from_formula
# ordinal_gee = gee_.OrdinalGEE.from_formula
# nominal_gee = gee_.NominalGEE.from_formula
# gee = gee_.GEE.from_formula
del lm_, dm_, glm_, roblm_  # , mlm_, qr_, hr_, gee_
