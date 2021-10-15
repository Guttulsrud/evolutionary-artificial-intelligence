def get_config_dict() -> dict:
    return {
        'independent_properties': {
            'time_steps': {
                'max': 10,
                'min': 10,
            },
            'kernel_size': {
                'values': [3],
            },
            'width': {
                'max': 10,
                'min': 10,
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
