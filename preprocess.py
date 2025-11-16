"""
dataimport.py - Tensor product operations on CSV data
"""
import numpy as np
import csv
'''
How to run:
from preprocess import TensorProductCalculator

calc = TensorProductCalculator('cov_1.csv', 'returns_1.csv')
calc.load_data()
calc.compute_tensor_products()
B, c = calc.get_results_as_lists()
'''

class TensorProductCalculator:
    """Class for reading CSV data and computing tensor products"""
    
    def __init__(self, vector_file=None, matrix_file=None):
        """
        Initialize with optional CSV filenames
        
        Args:
            vector_file: Path to CSV file containing vector b
            matrix_file: Path to CSV file containing matrix A
        """
        self.vector_file = vector_file
        self.matrix_file = matrix_file
        self.b = None
        self.A = None
        self.B = None
        self.c = None
        
    def read_vector_from_csv(self, filename):
        """Read a vector from CSV file (single column or row)"""
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        # Flatten and convert to float
        vector = [float(val) for row in data for val in row if val.strip()]
        return np.array(vector)
    
    def read_matrix_from_csv(self, filename):
        """Read a matrix from CSV file"""
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            data = [[float(val) for val in row if val.strip()] for row in reader if any(row)]
        return np.array(data)
    
    def load_data(self):
        """Load vector and matrix from CSV files"""
        if self.vector_file:
            self.b = self.read_vector_from_csv(self.vector_file)
        if self.matrix_file:
            self.A = self.read_matrix_from_csv(self.matrix_file)
    
    def set_data(self, b=None, A=None):
        """Manually set vector and/or matrix data"""
        if b is not None:
            self.b = np.array(b)
        if A is not None:
            self.A = np.array(A)
    
    def compute_tensor_products(self):
        """
        Compute tensor products:
        B = (A/4) ⊗ [[1, -1], [-1, 1]]
        c = (b/2) ⊗ [-1, 1]
        """
        if self.A is None or self.b is None:
            raise ValueError("Data not loaded. Use load_data() or set_data() first.")
        
        # Define the matrices from the formula
        matrix_1 = np.array([[1, -1], [-1, 1]])
        vector_1 = np.array([-1, 1])
        
        # Calculate B = (A/4) ⊗ [[1, -1], [-1, 1]]
        self.B = np.kron(self.A / 4, matrix_1)
        
        # Calculate c = (b/2) ⊗ [-1, 1]
        self.c = np.kron(self.b / 2, vector_1)
    
    def get_results_as_lists(self):
        """Return B and c as Python lists"""
        if self.B is None or self.c is None:
            raise ValueError("Results not computed. Run compute_tensor_products() first.")
        
        return self.B.tolist(), self.c.tolist()
    
    def print_results(self):
        """Print formatted results"""
        B_list, c_list = self.get_results_as_lists()
        
        print("Matrix B:")
        for row in B_list:
            print(row)
        print("\nVector c:")
        print(c_list)
