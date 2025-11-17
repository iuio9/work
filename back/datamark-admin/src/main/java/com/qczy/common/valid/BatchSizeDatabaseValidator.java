package com.qczy.common.valid;

import cn.hutool.core.util.ObjectUtil;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.qczy.common.annotation.ValidBatchSizeDatabase;
import com.qczy.model.entity.AlgorithmModelEntity;
import com.qczy.service.AlgorithmModelService;
import com.qczy.service.impl.AlgorithmModelServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;

import javax.validation.ConstraintValidator;
import javax.validation.ConstraintValidatorContext;

/**
 * @author ：gwj
 * @date ：Created in 2024-11-21 16:15
 * @description：
 * @modified By：
 * @version: $
 */
public class BatchSizeDatabaseValidator implements ConstraintValidator<ValidBatchSizeDatabase, String> {


    @Autowired
    AlgorithmModelService algorithmModelService;
    @Override
    public void initialize(ValidBatchSizeDatabase constraintAnnotation) {
        // Initialization code if necessary
    }

    @Override
    public boolean isValid(String value, ConstraintValidatorContext context) {
        if (value == null || value.isEmpty()) {
            return false; // Handle null/empty values (or let @NotNull handle it)
        }
        // Check if the batch size already exists in the database
        AlgorithmModelEntity one = algorithmModelService.getOne(new LambdaQueryWrapper<AlgorithmModelEntity>().eq(
                AlgorithmModelEntity::getModelName, value
        ));
        return ObjectUtil.isEmpty(one);
    }
}