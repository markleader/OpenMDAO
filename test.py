import openmdao.api as om
from openmdao.test_suite.components.paraboloid_distributed import DistParab
import numpy as np

# # cr = om.CaseReader('log_opt.sql_0')
# # print(sorted(cr.list_sources(out_stream=None)))
# # cases = cr.get_cases(source='driver')
# # for case in cases:
# #     case.list_outputs()

# # print('End')

size = 4

prob = om.Problem()
model = prob.model

ivc = om.IndepVarComp()
ivc.add_output('x', np.ones((size, )))
ivc.add_output('y', np.ones((size, )))
ivc.add_output('a', -3.0 + 0.6 * np.arange(size))

model.add_subsystem('p', ivc, promotes=['*'])
model.add_subsystem("parab", DistParab(arr_size=size, deriv_type='dense'), promotes=['*'])
model.add_subsystem('sum', om.ExecComp('f_sum = sum(f_xy)',
                                    f_sum=np.ones((size, )),
                                    f_xy=np.ones((size, ))),
                promotes=['*'])

model.add_design_var('x', lower=-50.0, upper=50.0)
model.add_design_var('y', lower=-50.0, upper=50.0)
model.add_objective('f_xy')
model.add_objective('f_sum', index=-1)


prob.driver = om.DOEDriver(om.FullFactorialGenerator(levels=2))
prob.driver.options['run_parallel'] = True
prob.driver.options['procs_per_model'] = 2

prob.driver.add_recorder(om.SqliteRecorder("cases.sql"))

prob.setup()
prob.run_driver()
prob.cleanup()


cr0 = om.CaseReader('cases.sql_0')

cr1 = om.CaseReader('cases.sql_1')

print("end")
