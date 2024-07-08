# RNN-MCTS-PINN
The code associated with our research paper titled: "RNN-MCTS-PINN: A Novel Integration of Recurrent Neural Networks, Monte Carlo Tree Search, and Physics-Informed Neural Networks for Symbolic Solutions to the Lane-Emden Type Equation"

To achieve the same results reported in this paper, please create an environment with the packages and their versions exactly mentioned in the `requirements.txt` file.

Each folder is associated with a certain case (m=0,1,2,3,4,5) of the standard Lane-Emden type equation. 

Each folder contains two Python files (`RNN_MCTS_M0x.py` and `PINN_Fine_Tune_M0x.py`) and a MATLAB file. The Python files are related to the algorithms and results reported in the paper. To get the same results as reported in the paper, we suggest you install the packages according to the `requirements.txt` file. Then run each case as follows:

1. For each case, first run the Python code `RNN_MCTS_M0x.py`. For example, for case m=0, please run `RNN_MCTS_M00.py`. This file will produce a file containing many valid branches resulting in a valid expression. Among these expressions, select the expression with the minimum loss value. Then copy that expression and the parameters into the associated `PINN_Fine_Tune_M0x.py` file. For example, for case m=0, the best record (the lowest loss) was as follows:

    """
    Expression= ((w(0)*(x*x))+w(1)*w(2))

    Parameters= tensor([-0.16138416528701782227, -0.98353004455566406250, -0.98389834165573120117])

    Loss= 0.0002932326460722834

    Main Expression in PyTorch= ((self.w[0]*(x*x))+self.w[1]*self.w[2])

    Branch= [1, 1, 3, 1, 3, 1, 3, 0, 3, 3, 0, 0, 0, 3, 3, 3, 3, 3]
    """

    You should copy `((w[0]*(x*x))+w[1]*w[2])` to variable `y` in `PINN_Fine_Tune_M00.py` file. Also, you need to copy `Parameters= tensor([-0.16138416528701782227, -0.98353004455566406250, -0.98389834165573120117])` to variable `w` in `PINN_Fine_Tune_M00.py`. Then run the file `PINN_Fine_Tune_M00.py`. This file will produce the fine-tuned parameters `w`.

2. The MATLAB file in each folder holds the visualization responsibility.
