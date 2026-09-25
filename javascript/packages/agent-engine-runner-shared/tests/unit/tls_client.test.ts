/**
 * Unit tests for tls_client module (HTTPS mTLS configuration).
 *
 * Tests both the happy path and fail-closed error paths to ensure HTTPS
 * misconfiguration is detected at client creation time, not at request time.
 *
 * Mirrors Python's test_tls_client.py coverage.
 */

import {
  ROOT_CONTEXT,
  TraceFlags,
  context,
  propagation,
  trace,
} from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  fetchPlatform,
  getFetchOptionsWithTLS,
  closeAllTLSAgents,
  TLS_AGENT_CACHE_MAX_ENTRIES,
} from "../../src/tls_client.js";

describe("tls_client", () => {
  let tempDir: string;
  const originalEnv = { ...process.env };

  beforeEach(() => {
    // Create temp directory for cert files
    tempDir = mkdtempSync(join(tmpdir(), "tls-test-"));

    // Clear TLS env vars
    delete process.env.TLS_CERT_PATH;
    delete process.env.TLS_KEY_PATH;
    delete process.env.TLS_CA_CERT_PATH;
    delete process.env.TLS_CERT_PEM;
    delete process.env.TLS_KEY_PEM;
    delete process.env.TLS_CA_CERT_PEM;

    // Close any cached agents from previous tests
    closeAllTLSAgents();
  });

  afterEach(() => {
    // Clean up temp directory
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }

    // Restore env
    process.env = { ...originalEnv };

    // Close agents
    closeAllTLSAgents();
    vi.restoreAllMocks();
  });

  describe("platform trace propagation", () => {
    it("injects trace context without baggage and suppresses transport tracing", async () => {
      const activeContext = propagation.setBaggage(
        trace.setSpanContext(ROOT_CONTEXT, {
          traceId: "0123456789abcdef0123456789abcdef",
          spanId: "0123456789abcdef",
          traceFlags: TraceFlags.SAMPLED,
        }),
        propagation.createBaggage({
          secret: { value: "must-not-cross-service-boundary" },
        }),
      );
      vi.spyOn(context, "active").mockReturnValue(activeContext);
      const withSpy = vi.spyOn(context, "with");
      const fetchMock = vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 204 }));

      await fetchPlatform(
        "http://oe.test/tool/execute",
        { method: "POST", headers: { "X-Test": "value" } },
        fetchMock,
      );

      const init = fetchMock.mock.calls.at(0)?.[1];
      expect(init).toBeDefined();
      const headers = (init?.headers ?? {}) as Record<string, string>;
      expect(headers).toMatchObject({
        "X-Test": "value",
        traceparent: "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
      });
      expect(headers).not.toHaveProperty("baggage");
      const suppressedContext = withSpy.mock.calls.at(0)?.[0];
      expect(
        suppressedContext ? isTracingSuppressed(suppressedContext) : false,
      ).toBe(true);
    });
  });

  describe("HTTP passthrough", () => {
    it("returns undefined for http:// URLs", () => {
      const result = getFetchOptionsWithTLS("http://localhost:8000");
      expect(result).toBeUndefined();
    });

    it("returns undefined for http:// URLs even with TLS env vars set", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      writeFileSync(certPath, "fake-cert");
      writeFileSync(keyPath, "fake-key");
      writeFileSync(caPath, "fake-ca");

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      const result = getFetchOptionsWithTLS("http://localhost:8000");
      expect(result).toBeUndefined();
    });
  });

  describe("HTTPS fail-closed behavior", () => {
    it("throws when https:// URL used without any TLS env vars", () => {
      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });

    it("throws when https:// URL has only TLS_CERT_PATH", () => {
      const certPath = join(tempDir, "cert.crt");
      writeFileSync(certPath, "fake-cert");
      process.env.TLS_CERT_PATH = certPath;

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });

    it("throws when https:// URL has only TLS_KEY_PATH", () => {
      const keyPath = join(tempDir, "key.key");
      writeFileSync(keyPath, "fake-key");
      process.env.TLS_KEY_PATH = keyPath;

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });

    it("throws when https:// URL has only TLS_CA_CERT_PATH", () => {
      const caPath = join(tempDir, "ca.crt");
      writeFileSync(caPath, "fake-ca");
      process.env.TLS_CA_CERT_PATH = caPath;

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });

    it("throws when https:// URL has cert+key but no CA", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");

      writeFileSync(certPath, "fake-cert");
      writeFileSync(keyPath, "fake-key");

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });
  });

  describe("Valid PATH mode (container)", () => {
    it("returns dispatcher with Undici agent for https:// with all certs", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      // Write minimal valid PEM content
      writeFileSync(
        certPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );
      writeFileSync(
        keyPath,
        "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n-----END PRIVATE KEY-----\n",
      );
      writeFileSync(
        caPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      const result = getFetchOptionsWithTLS("https://oe-service:8443");

      expect(result).toBeDefined();
      expect(result).toHaveProperty("dispatcher");
      expect(result?.dispatcher).toBeDefined();
    });

    it("throws when TLS_CERT_PATH points to nonexistent file", () => {
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      writeFileSync(keyPath, "fake-key");
      writeFileSync(caPath, "fake-ca");

      process.env.TLS_CERT_PATH = join(tempDir, "nonexistent.crt");
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow();
    });
  });

  describe("Valid PEM mode (VM)", () => {
    it("returns dispatcher when all TLS_*_PEM vars are set", () => {
      process.env.TLS_CERT_PEM =
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n";
      process.env.TLS_KEY_PEM =
        "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n-----END PRIVATE KEY-----\n";
      process.env.TLS_CA_CERT_PEM =
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n";

      const result = getFetchOptionsWithTLS("https://oe-service:8443");

      expect(result).toBeDefined();
      expect(result).toHaveProperty("dispatcher");
      expect(result?.dispatcher).toBeDefined();
    });

    it("throws when PEM vars are empty strings", () => {
      process.env.TLS_CERT_PEM = "";
      process.env.TLS_KEY_PEM = "";
      process.env.TLS_CA_CERT_PEM = "";

      expect(() => {
        getFetchOptionsWithTLS("https://oe-service:8443");
      }).toThrow(/HTTPS URL requires mTLS configuration/);
    });
  });

  describe("Agent caching per host", () => {
    it("reuses agent for same base URL", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      writeFileSync(
        certPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );
      writeFileSync(
        keyPath,
        "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n-----END PRIVATE KEY-----\n",
      );
      writeFileSync(
        caPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      const result1 = getFetchOptionsWithTLS("https://oe-service:8443");
      const result2 = getFetchOptionsWithTLS(
        "https://oe-service:8443/some/path",
      );

      // Same base URL should return same agent instance
      expect(result1?.dispatcher).toBe(result2?.dispatcher);
    });

    it("creates different agents for different hosts", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      writeFileSync(
        certPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );
      writeFileSync(
        keyPath,
        "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n-----END PRIVATE KEY-----\n",
      );
      writeFileSync(
        caPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      const result1 = getFetchOptionsWithTLS("https://oe-service:8443");
      const result2 = getFetchOptionsWithTLS("https://other-service:8443");

      // Different hosts should have different agent instances
      expect(result1?.dispatcher).not.toBe(result2?.dispatcher);
    });

    it("evicts least-recently-used replica origins", () => {
      process.env.TLS_CERT_PEM = "test-cert";
      process.env.TLS_KEY_PEM = "test-key";
      process.env.TLS_CA_CERT_PEM = "test-ca";

      const first = getFetchOptionsWithTLS("https://owner-0.oe-headless:8443");
      for (let index = 1; index <= TLS_AGENT_CACHE_MAX_ENTRIES; index += 1) {
        getFetchOptionsWithTLS(`https://owner-${index}.oe-headless:8443`);
      }

      const recreated = getFetchOptionsWithTLS(
        "https://owner-0.oe-headless:8443",
      );
      expect(recreated?.dispatcher).not.toBe(first?.dispatcher);
    });
  });

  describe("closeAllTLSAgents", () => {
    it("closes all cached agents", () => {
      const certPath = join(tempDir, "cert.crt");
      const keyPath = join(tempDir, "key.key");
      const caPath = join(tempDir, "ca.crt");

      writeFileSync(
        certPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );
      writeFileSync(
        keyPath,
        "-----BEGIN PRIVATE KEY-----\nMIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQg\n-----END PRIVATE KEY-----\n",
      );
      writeFileSync(
        caPath,
        "-----BEGIN CERTIFICATE-----\nMIIBkTCB+wIJAKHHCgVZU1p0MA0GCSqGSIb3DQEBCwUAMA0xCzAJBgNVBAYTAlVT\n-----END CERTIFICATE-----\n",
      );

      process.env.TLS_CERT_PATH = certPath;
      process.env.TLS_KEY_PATH = keyPath;
      process.env.TLS_CA_CERT_PATH = caPath;

      // Create multiple agents
      getFetchOptionsWithTLS("https://oe-service:8443");
      getFetchOptionsWithTLS("https://other-service:8443");

      // Should not throw
      expect(() => closeAllTLSAgents()).not.toThrow();

      // After closing, new requests should create fresh agents
      const result = getFetchOptionsWithTLS("https://oe-service:8443");
      expect(result).toBeDefined();
      expect(result?.dispatcher).toBeDefined();
    });
  });
});
