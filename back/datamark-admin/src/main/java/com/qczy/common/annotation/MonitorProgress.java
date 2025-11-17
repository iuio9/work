package com.qczy.common.annotation;

/**
 * @author ：gwj
 * @date ：Created in 2024-08-24 15:37
 * @description：
 * @modified By：
 * @version: $
 */
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
public @interface MonitorProgress {
}
