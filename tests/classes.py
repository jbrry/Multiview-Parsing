
class MyNet():
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, letters=["a", "b", "c", "d", "e"]):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)



        self.letters = letters
        self.cool_dict = {}
        for letter in self.letters:
            self.cool_dict[letter] = letters
        #super(MyNet, self).__init__(params, defaults)



class Debugger():

    def __init__(self):
        self.mn = MyNet()

        print(self.mn.letters)
        print(self.mn.cool_dict)

        #self.mn.letters = ['a', 'a', 'a', 'a', 'a']

        self.mn = MyNet(letters=['a', 'a', 'a', 'a', 'a'])
        print(self.mn.letters)
        print(self.mn.cool_dict)


        #mn.letters = ['a', 'a', 'a', 'a', 'a']
        #print(mn.cool_dict)

# 2) Check that the parameters of the sub-optimizers were also changed:
# This is purely just for debugging purposes.
for name, optimizer in self.optimizer._grouped_optimizers.items():
    for param_group in optimizer.param_groups:
        for p in param_group["params"]:
            assert p.grad == None

x = Debugger()