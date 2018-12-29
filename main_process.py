import decision_tree_id3 as id3
import decision_tree_c45 as c45
import decision_tree_cart as cart

if __name__ == '__main__':
    print("******************Classification ID3 Algorithm******************")
    id3.dt_id3()
    print("******************Classification C4.5 Algorithm******************")
    c45.dt_c45()
    print("******************Classification CART Algorithm******************")
    cart.dt_cart()
