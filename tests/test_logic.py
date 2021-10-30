import unittest


from igre.logic import *


class TestSentenceParsing(unittest.TestCase):

    def parse(self,snt_str):
        return LogicSentenceParser()(snt_str)
        
    def test_atomic_sentence(self):

        self.assertEqual(self.parse("right(u1,u2)"),AtomicSentence("right", ["u1","u2"]))
        self.assertEqual(self.parse("red(u1)"),AtomicSentence("red", ["u1"]))
        self.assertEqual(self.parse("red(x1)"),AtomicSentence("red", ["x1"]))

    def test_connective_sentence(self):

        self.assertEqual(self.parse("red(u1) ^ blue(u3)"),AndSentence(AtomicSentence("red",["u1"]),AtomicSentence("blue",["u3"])))
        self.assertEqual(self.parse("red(u1) v blue(u3)"),OrSentence(AtomicSentence("red",["u1"]),AtomicSentence("blue",["u3"])))
        self.assertEqual(self.parse("red(u1) => blue(u3)"),ImpSentence(AtomicSentence("red",["u1"]),AtomicSentence("blue",["u3"])))
        self.assertEqual(self.parse("red(u1) <=> blue(u3)"),IffSentence(AtomicSentence("red",["u1"]),AtomicSentence("blue",["u3"])))

    def test_negated_sentence(self):

        self.assertEqual(self.parse("!(red(u1))"),NegatedSentence(AtomicSentence("red", ["u1"])))
        self.assertEqual(self.parse("!(red(u1) ^ blue(u3))"),NegatedSentence(AndSentence(AtomicSentence("red",["u1"]),AtomicSentence("blue",["u3"]))))

class TestRefExpParsing(unittest.TestCase):

    def parse(self,ref_exp_str):
        return LogicRefExpParser()(ref_exp_str)

    def test_parsing(self):
        self.assertEqual(str(self.parse("< _a_q x1. blue(x1) ^ square(x1) >").snt),"blue(x1) ^ square(x1)")


# model_one = DomainModel(domain={"u1","u2","u3"},valuation={
#         "red": {"u1","u2"},
#         "blue": {"u3"},
#         "square":{"u1","u2","u3"},
#         "right":{("u1","u2")}
#     })

# # snt string, snt object, snt evaluation 
# snt_one = [
#     ("right(u1,u2)",
#     AtomicSentence("right", ["u1","u2"]),
#     True),
#     ("red(u1)",
#     AtomicSentence("red", ["u1"]),
#     True),
#     ("red(u3) ^ square(u3)",
#     AndSentence(left=AtomicSentence("blue", ["u3"]),
#                 right=AtomicSentence("red", ["u3"]),),
#     False),
#     ("(blue(u3) v red(u3)) ^ square(u3)",
#     AndSentence(OrSentence(left=AtomicSentence("blue", ["u3"]),
#                         right=AtomicSentence("red", ["u3"])),
#                 AtomicSentence("square",["u3"])),
#     True),
# ]

# model_two = DomainModel(domain={"u1","u2","u3","u4"},valuation={
#         "red": {"u4","u2"},
#         "green": {"u3","u1"},
#         "square":{"u1","u2","u3"},
#         "thing":{"u1","u2","u3","u4"},
#         "left":{("u2","u1"),("u2","u3")},
#     })


# @pytest.mark.parametrize("snt_str, snt, evaluation",snt_one)
# def test_situation_1(snt_str, snt, evaluation)->None:
#     assert LogicSentenceParser()(snt_str) == snt
#     # assert model_one.eval(LogicSentenceParser()(snt), g={}) == evaluation




# @pytest.mark.parametrize("snt_str, snt, evaluation",snt_one)
# @pytest.mark.parametrize("name",[ "_a_q",
#                                 "_every_q",
#                                 "_both_q",
#                                 "_exactly_1_q",
#                                 "_more_than_2_q",
#                                 "_less_than_3_q",
#                                 "_at_most_4_q",
#                                 "_at_least_5_q",
#                                 "_the_6_q",
#                                 "_all_but_7_q",
#                                 "_1_of_the_2_q",
#                         ])
# def test_ref_exp_parsing(snt_str, snt, name)->None:



# # snt_eval_factory_model_0 = [
# # ("_every_q x1.(red(x1), square(x1))", True),
# # ("_a_q x1.(blue(x1), square(x1))", True),
# # ("_1_of_the_2_q x1.(red(x1), square(x1))", False),
# # ]

# @pytest.mark.parametrize("snt_str, snt, evaluation",snt_one)
# def test_model_one_eval(snt, eval):
#     assert model_one.eval(LogicSentenceParser()(snt), g={}) == eval


# snt_eval_factory_model_1 = [
# ("_a_q x1.(red(x1), green(x1))", False),
# ("_every_q x1.(thing(x1), square(x1))", False),
# ("_exactly_1_q x1.(red(x1) ^ thing(x1), _both_q x2.(green(x2) ^ square(x2), left(x1,x2)))", True),
# ("_more_than_1_q x1.(red(x1) ^ thing(x1), _both_q x2.(green(x2) ^ square(x2), left(x1,x2)))", False),
# ]
# @pytest.mark.parametrize("snt, eval",snt_eval_factory_model_1)
# def test_domain_model_eval_1(snt, eval):
#     assert model_factory[1].eval(LogicSentenceParser()(snt), g={}) == eval


# ref_exp_referent_sentence_factory = [
#     ("< _a_q x1. blue(x1) ^ square(x1) >", frozenset([frozenset(["u3"])]), "blue(u3) ^ square(u3)"),
#     ("< _exactly_2_q x1. red(x1) ^ square(x1) >", frozenset([frozenset(["u1","u2"])]), "red(u1) ^ square(u1) ^ red(u2) ^ square(u2)" ),
#     ("< _the_2_q x1. red(x1) ^ square(x1) >", frozenset([frozenset(["u1","u2"])]), "red(u1) ^ square(u1) ^ red(u2) ^ square(u2) ^ !(red(u3) ^ square(u3))" ),
# ]
# @pytest.mark.parametrize("refexp, referent, snt_str",ref_exp_referent_sentence_factory )
# def test_domain_model_ref_0(refexp, referent, snt_str):
#     assert model_factory[0].referent(LogicRefExpParser()(refexp)) == referent






# # @pytest.mark.parametrize("ref_exp, referent, snt_str", ref_exp_referent_sentence_factory)
# # def test_ref_exp2sentence(ref_exp, referent, snt_str):
#     snt = refexp2sentence(LogicRefExpParser()(ref_exp),referent, model_factory[0])
#     assert str(snt) == snt_str

# snt_ground_snt_factory = [
#     ("red(u1)",)
# ]

# @pytest.mark.parametrize("snt,ground_snt", snt_ground_snt_factory)
# @pytest.mark.parametrize("model", [model_factory[0]])
# def test_snt_to_ground_snt(snt, model, ground_snt):
#     assert  snt(grounding(snt, model)) == str(ground_snt) 

# sentence_factory = [
#     ("right(x1,u2)",AtomicSentence(name="right", terms=["x1","u2"])),
#     ("red(x4)",AtomicSentence(name="red", terms=["x4"])),
#     ("red(u1)",AtomicSentence(name="red", terms=["u1"])),
#     ("blue(x14) ^ red(x4)", AndSentence(
#         left=AtomicSentence("blue", ["x14"]),
#         right=AtomicSentence("red", ["x4"]),
#     ) ),
#     ("blue(x14) v red(x4)", OrSentence(
#         left=AtomicSentence("blue", ["x14"]),
#         right=AtomicSentence("red", ["x4"]),
#     ) ),
#     ("blue(x14) => red(x4)", ImpSentence(
#         left=AtomicSentence("blue", ["x14"]),
#         right=AtomicSentence("red", ["x4"]),
#     ) ),
#     ("blue(x14) <=> red(x4)", IffSentence(
#         left=AtomicSentence("blue", ["x14"]),
#         right=AtomicSentence("red", ["x4"]),
#     ) ),
#     ("_a_q x4.(blue(x4),square(x4))", QuantifierSentence(
#         name="_a_q",
#         var =Variable("x4"),
#         rstr=AtomicSentence("blue", ["x14"]),
#         body=AtomicSentence("red", ["x4"]),
#     ) ),  
#     ("_a_q x4.(blue(x4)^right(u1,x4),square(x4))", QuantifierSentence(
#         name="_a_q",
#         var =Variable("x4"),
#         rstr=AndSentence(
#             left=AtomicSentence("blue", ["x4"]),
#             right=AtomicSentence("right", ["u1","x4"])),
#         body=AtomicSentence("red", ["x4"]),
#     ) ),  
# ]
