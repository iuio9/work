package com.qczy.utils;

public class SnowflakeIdGenerator {

    private final static long START_STAMP = 1581818800000L; // 2020-02-16 00:00:00

    private long lastStamp = -1L;
    private long sequence = 0L;

    private final static long MACHINE_BIT = 10;
    private final static long MAX_SEQUENCE = ~(-1L << MACHINE_BIT);
    private final static long MACHINE_ID = 1L; // 机器ID，可以根据实际情况设置

    public synchronized long generateId() {
        long currentStamp = System.currentTimeMillis();
        if (currentStamp < lastStamp) {
            throw new RuntimeException("Clock moved backwards. Refusing to generate id");
        }

        if (currentStamp == lastStamp) {
            sequence = (sequence + 1) & MAX_SEQUENCE;
            if (sequence == 0) {
                currentStamp = getNextMill();
            }
        } else {
            sequence = 0L;
        }

        lastStamp = currentStamp;

        return ((currentStamp - START_STAMP) << (MACHINE_BIT + 12)) | (MACHINE_ID << 12) | sequence;
    }

    private long getNextMill() {
        long mill = System.currentTimeMillis();
        while (mill <= lastStamp) {
            mill = System.currentTimeMillis();
        }
        return mill;
    }

    public static void main(String[] args) {
        SnowflakeIdGenerator idGenerator = new SnowflakeIdGenerator();
        System.out.println(idGenerator.generateId());
    }
}