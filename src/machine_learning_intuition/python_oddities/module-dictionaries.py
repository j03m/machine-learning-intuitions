from machine_learning_intuition.python_oddities import config


print("treating module as dict lets us use values as strings:", config.__dict__["I_AM_SOME_VALUE"])


print("vs dot:", config.I_AM_SOME_VALUE)

print("but this is not allowed:")

try:
    print(config["I_AM_SOME_VALUE"])
except TypeError as e:
    print("You get: ", e)

print("which I found odd...so here we are in python oddities")