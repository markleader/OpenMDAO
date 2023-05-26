import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_totals
from openmdao.test_suite.components.exec_comp_for_test import ExecComp4Test


class TestPrePostIter(unittest.TestCase):

    def setup_problem(self, do_pre_post_opt, use_ivc=False, coloring=False, size=3):
        prob = om.Problem()
        prob.options['group_by_pre_opt_post'] = do_pre_post_opt

        prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', disp=False)
        prob.set_solver_print(level=0)

        model = prob.model

        model.add_subsystem('pre1', ExecComp4Test('y=2.*x', x=np.ones(size), y=np.zeros(size)))
        model.add_subsystem('pre2', ExecComp4Test('y=3.*x', x=np.ones(size), y=np.zeros(size)))

        if use_ivc:
            model.add_subsystem('ivc', om.IndepVarComp('x', np.ones(size)))
        model.add_subsystem('iter1', ExecComp4Test('y=x1 + x2*4. + x3',
                                                   x1=np.ones(size), x2=np.ones(size),
                                                   x3=np.ones(size), y=np.zeros(size)))
        model.add_subsystem('iter2', ExecComp4Test('y=.5*x', x=np.ones(size), y=np.zeros(size)))
        model.add_subsystem('iter4', ExecComp4Test('y=7.*x', x=np.ones(size), y=np.zeros(size)))
        model.add_subsystem('iter3', ExecComp4Test('y=6.*x', x=np.ones(size), y=np.zeros(size)))

        model.add_subsystem('post1', ExecComp4Test('y=8.*x', x=np.ones(size), y=np.zeros(size)))
        model.add_subsystem('post2', ExecComp4Test('y=x1*9. + x2*5', x1=np.ones(size),
                                                   x2=np.ones(size), y=np.zeros(size)))

        # we don't want ExecComps to be colored because it makes the iter counting more complicated
        for comp in model.system_iter(typ=ExecComp4Test):
            comp.options['do_coloring'] = False
            comp.options['has_diag_partials'] = True

        if use_ivc:
            model.connect('ivc.x', 'iter1.x3')
        model.connect('pre1.y', ['iter1.x1', 'post2.x1'])
        model.connect('pre2.y', 'iter1.x2')
        model.connect('iter1.y', ['iter2.x', 'iter4.x'])
        model.connect('iter2.y', 'post2.x2')
        model.connect('iter3.y', 'post1.x')
        model.connect('iter4.y', 'iter3.x')

        prob.model.add_design_var('iter1.x3', lower=0, upper=10)
        prob.model.add_constraint('iter2.y', upper=10.)
        prob.model.add_objective('iter3.y', index=0)

        if coloring:
            prob.driver.declare_coloring()

        return prob

    def test_pre_post_iter_rev(self):
        prob = self.setup_problem(do_pre_post_opt=True)
        prob.setup(mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, coloring=True)
        prob.setup(mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_ivc(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True)
        prob.setup(mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_rev_ivc_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True)
        prob.setup(mode='rev')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd(self):
        prob = self.setup_problem(do_pre_post_opt=True)
        prob.setup(mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, coloring=True)
        prob.setup(mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_ivc(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True)
        prob.setup(mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_fwd_ivc_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True)
        prob.setup(mode='fwd')
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 3)
        self.assertEqual(prob.model.iter2.num_nl_solves, 3)
        self.assertEqual(prob.model.iter3.num_nl_solves, 3)
        self.assertEqual(prob.model.iter4.num_nl_solves, 3)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx(self):
        prob = self.setup_problem(do_pre_post_opt=True)
        prob.model.approx_totals()

        prob.setup(mode='fwd', force_alloc_complex=True)
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 9)
        self.assertEqual(prob.model.iter2.num_nl_solves, 9)
        self.assertEqual(prob.model.iter3.num_nl_solves, 9)
        self.assertEqual(prob.model.iter4.num_nl_solves, 9)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, coloring=True)
        prob.model.approx_totals()

        prob.setup(mode='fwd', force_alloc_complex=True)
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 16)
        self.assertEqual(prob.model.iter2.num_nl_solves, 16)
        self.assertEqual(prob.model.iter3.num_nl_solves, 16)
        self.assertEqual(prob.model.iter4.num_nl_solves, 16)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_ivc(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True)
        prob.model.approx_totals()

        prob.setup(mode='fwd', force_alloc_complex=True)
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 9)
        self.assertEqual(prob.model.iter2.num_nl_solves, 9)
        self.assertEqual(prob.model.iter3.num_nl_solves, 9)
        self.assertEqual(prob.model.iter4.num_nl_solves, 9)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)

    def test_pre_post_iter_approx_ivc_coloring(self):
        prob = self.setup_problem(do_pre_post_opt=True, use_ivc=True, coloring=True)
        prob.model.approx_totals()

        prob.setup(mode='fwd', force_alloc_complex=True)
        prob.run_driver()

        self.assertEqual(prob.model.pre1.num_nl_solves, 1)
        self.assertEqual(prob.model.pre2.num_nl_solves, 1)

        self.assertEqual(prob.model.iter1.num_nl_solves, 16)
        self.assertEqual(prob.model.iter2.num_nl_solves, 16)
        self.assertEqual(prob.model.iter3.num_nl_solves, 16)
        self.assertEqual(prob.model.iter4.num_nl_solves, 16)

        self.assertEqual(prob.model.post1.num_nl_solves, 1)
        self.assertEqual(prob.model.post2.num_nl_solves, 1)

        data = prob.check_totals(method='cs', out_stream=None)
        assert_check_totals(data)


if __name__ == "__main__":
    unittest.main()
