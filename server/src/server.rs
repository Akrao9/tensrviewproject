//! Listening, the Bearer-token check, and the rule for binding beyond loopback.

use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::service::interceptor::InterceptedService;
use tonic::transport::{Identity, Server, ServerTlsConfig};
use tonic::{Request, Status};

use crate::ingest::{Ingest, IngestService};
use crate::storage::Error;

/// A batch is at most 4 MiB of JSON lines on the sender; leave room for wide rows.
const MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

/// A server that anyone on the network can reach must check a token, unless the operator
/// says otherwise.
pub fn check_bind(address: &SocketAddr, has_token: bool, allow_unauthenticated: bool) -> Result<(), String> {
    if address.ip().is_loopback() || has_token || allow_unauthenticated {
        return Ok(());
    }
    Err(format!(
        "refusing to listen on {address} without authentication: set TENSORLENS_TOKEN, \
         or pass --allow-unauthenticated, or listen on 127.0.0.1"
    ))
}

/// Compares every byte, so the time taken does not reveal how much of a guess was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut difference = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        difference |= (a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)) as usize;
    }
    difference == 0
}

fn check_token(request: Request<()>, token: Option<&str>) -> Result<Request<()>, Status> {
    let Some(token) = token else { return Ok(request) };
    let expected = format!("Bearer {token}");
    let given = request
        .metadata()
        .get("authorization")
        .map(|value| value.as_bytes())
        .unwrap_or_default();
    if constant_time_eq(given, expected.as_bytes()) {
        Ok(request)
    } else {
        Err(Status::unauthenticated("missing or wrong Bearer token"))
    }
}

pub struct Tls {
    pub certificate_pem: Vec<u8>,
    pub key_pem: Vec<u8>,
}

/// Serves `DoPut` on `listener` until `shutdown` resolves.
pub async fn serve(
    listener: TcpListener,
    ingest: Arc<Ingest>,
    token: Option<String>,
    tls: Option<Tls>,
    shutdown: impl Future<Output = ()>,
) -> Result<(), Error> {
    let service = FlightServiceServer::new(IngestService::new(ingest)).max_decoding_message_size(MAX_MESSAGE_BYTES);
    let service = InterceptedService::new(service, move |request| check_token(request, token.as_deref()));
    let mut builder = Server::builder().tcp_nodelay(true).http2_adaptive_window(Some(true));
    if let Some(tls) = tls {
        let identity = Identity::from_pem(tls.certificate_pem, tls.key_pem);
        builder = builder.tls_config(ServerTlsConfig::new().identity(identity))?;
    }
    builder
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_loopback_bind_needs_a_token_or_an_explicit_flag() {
        let public: SocketAddr = "0.0.0.0:50051".parse().unwrap();
        let loopback: SocketAddr = "127.0.0.1:50051".parse().unwrap();
        let loopback6: SocketAddr = "[::1]:50051".parse().unwrap();
        assert!(check_bind(&public, false, false).is_err());
        assert!(check_bind(&public, true, false).is_ok());
        assert!(check_bind(&public, false, true).is_ok());
        assert!(check_bind(&loopback, false, false).is_ok());
        assert!(check_bind(&loopback6, false, false).is_ok());
    }

    #[test]
    fn token_comparison() {
        assert!(constant_time_eq(b"Bearer abc", b"Bearer abc"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abd"));
        assert!(!constant_time_eq(b"Bearer abc", b"Bearer abcd"));
        assert!(!constant_time_eq(b"", b"Bearer abc"));
    }
}
