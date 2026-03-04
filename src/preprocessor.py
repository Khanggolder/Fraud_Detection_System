#preprocessor.py
import ast
#khoan dung AST, dựa vào khoảng trắng để tính điểm, Nếu có thì trừ, trích xuất cmt (Feature Engineering), nhiều style viết code, tạo dataset
class ASTNormalizer(ast.NodeTransformer):
    def __init__(self):
        self.var_counter = 0
        self.func_counter = 0
        self.var_map = {}
        self.func_map = {}

    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Str, ast.Constant)):
            node.body.pop(0)
        
        if node.name not in self.func_map:
            self.func_counter += 1
            new_name = f"func_{self.func_counter}"
            self.func_map[node.name] = new_name
        
        node.name = self.func_map[node.name]
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)):
            if node.id not in self.var_map and node.id not in self.func_map:
                 if not node.id.startswith('__'):
                    self.var_counter += 1
                    new_name = f"var_{self.var_counter}"
                    self.var_map[node.id] = new_name
            
            if node.id in self.var_map:
                node.id = self.var_map[node.id]
            elif node.id in self.func_map:
                node.id = self.func_map[node.id]
        
        return node

    def visit_arg(self, node):
        if node.arg not in self.var_map:
            self.var_counter += 1
            new_name = f"var_{self.var_counter}"
            self.var_map[node.arg] = new_name
        node.arg = self.var_map[node.arg]
        return node

def preprocess_code(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""

    normalizer = ASTNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    return ast.unparse(normalized_tree)
