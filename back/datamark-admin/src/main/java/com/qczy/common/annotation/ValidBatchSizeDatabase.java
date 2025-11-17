package com.qczy.common.annotation;

import com.qczy.common.valid.BatchSizeDatabaseValidator;

import javax.validation.Constraint;
import javax.validation.Payload;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * @author ：gwj
 * @date ：Created in 2024-11-21 16:13
 * @description：
 * @modified By：
 * @version: $
 */
@Target({ ElementType.FIELD }) // Applicable to fields
@Retention(RetentionPolicy.RUNTIME) // Retain at runtime
@Constraint(validatedBy = BatchSizeDatabaseValidator.class) // Link to custom validator
public @interface ValidBatchSizeDatabase {
    String message() default "该模型已存在,请重新命名任务名称";

    Class<?>[] groups() default {};

    Class<? extends Payload>[] payload() default {};
}