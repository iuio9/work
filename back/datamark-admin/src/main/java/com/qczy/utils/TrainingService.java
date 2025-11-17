package com.qczy.utils;

import com.qczy.model.entity.TrainingParams;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.validation.annotation.Validated;

import javax.validation.ConstraintViolation;
import javax.validation.Valid;
import javax.validation.Validator;
import java.util.Set;

/**
 * @author ：gwj
 * @date ：Created in 2024-11-15 14:20
 * @description：
 * @modified By：
 * @version: $
 */
@Service
@Validated
public class TrainingService {
    @Autowired
    private Validator validator;



    public String validateTrainingParams(TrainingParams params) {
        Set<ConstraintViolation<TrainingParams>> violations = validator.validate(params);
        if (!violations.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (ConstraintViolation<TrainingParams> violation : violations) {
                sb.append(violation.getMessage()).append("; ");
            }
            throw new IllegalArgumentException("Validation failed: " + sb.toString());
        }
        return "Validation Successful!";
    }
}
