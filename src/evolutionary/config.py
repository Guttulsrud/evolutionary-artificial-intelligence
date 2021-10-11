def get_config_dict() -> dict:
    return {
        'independent_properties': {
            'time_steps': {
                'max': 100,
                'min': 1,
            },
            'kernel_size': {
                'values': [3, 5],
            },
            'width': {
                'max': 100,
                'min': 1,
            },
        },
        'dependent_properties': {
            'inputs': [{
                'name': 'angle_index'
            }, ],
            'output': [{
                'name': 'action_index'
            }, ],
            'rule_number': {
                'max': None,
                'min': 0,
            },
        }
        # 'boundary_condition': 'periodic'
    }
